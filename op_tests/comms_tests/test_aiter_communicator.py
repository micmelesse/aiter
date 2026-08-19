# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Does each communicator backend work?

One test per (backend, mode). `exercise` puts ONE communicator through the whole API -- both
collectives, both dtypes, every shape -- because that is what vLLM does. Every cell is four steps:
gen_inputs -> run_collective -> expected_outputs -> compare.

The modes localise a failure rather than covering different ground: `eager` asks whether the
collective is right at all, `graph` whether capture/replay preserves that across many replays with
fresh input, `vllm` whether the real pattern works (one collective per layer in one capture).
"""

import logging
from functools import partial
from itertools import product
from dataclasses import dataclass
import multiprocessing as mp
import time
from multiprocessing import Pool, set_start_method
from typing import Tuple

import pytest
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from aiter.dist.utils import get_distributed_init_method, get_open_port
from aiter.ops.triton.comms.communicator import (
    HipCommunicator,
    IrisCommunicator,  # noqa: F401 -- re-enabled by uncommenting it below
    TorchCommunicator,
    make_communicator,
)

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

# Replays for the `graph` case. Back-to-back with no inter-replay sync is what stresses an
# elided end barrier (replay N+1 must not start before replay N's writes land); a per-replay sync
# would hide the race. 200 rather than more because each replay also keeps a snapshot output buffer
# (N x out-shape) and a stale read shows within the first few differing replays anyway.
GRAPH_REPLAYS = 200


CASE_TIMEOUT_S = 600

# Deterministic per-(rank, replay) seed base for the varying-input check.
_INPUT_SEED = 20260615

# The admission parameters CustomAllreduce uses, RECORDED here. That is the path these backends
# replace, so it is the envelope they have to match, and writing the numbers down is what makes the
# match reviewable. aiter must not IMPORT vllm to get them -- vllm depends on aiter, not the reverse.
#
# From vllm/distributed/device_communicators/custom_all_reduce.py, `should_custom_ar`:
#     inp_size % 16 == 0, is_weak_contiguous(inp), and inp_size < self.max_size (default 8 MiB).
#
# NOT from `aiter/dist/device_communicators/custom_all_reduce.py`, which is a modified fork: it
# hardcodes `fully_connected`, splits the bound with an `8192*8192` constant, and adds a
# `should_custom_ag` upstream does not have. Reading it as the reference produced two wrong bounds.
BASELINE_MAX_SIZE = 8 * 1024 * 1024      # CustomAllreduce's default max_size
BASELINE_ALIGNMENT = 16                  # "input byte size to be multiples of 16"


# THE COMMUNICATOR'S API, and the test is a sweep over it. A name is both the collective to call and,
# with the underscores dropped, the gate to ask (`all_reduce` / `should_allreduce`), so `getattr` needs
# no table -- and renaming the API breaks this loudly instead of testing something else quietly.
OPS = ("all_reduce", "all_gather")


def _say(rank, line):
    """Print from rank 0 only: eight ranks saying the same thing is one fact, eight times."""
    if rank == 0:
        print(line, flush=True)


def _expected(op_name, inputs):
    """What every rank should hold after `op_name`. The ONE thing that genuinely differs per
    collective and is not derivable from the API: what the right answer IS.

    all_reduce sums, accumulated in fp32 so the reference does not itself eat bf16 rounding.
    all_gather concatenates rank-ordered along the last axis -- the `Communicator.all_gather`
    contract, identical for every backend.
    """
    if op_name == "all_reduce":
        acc = torch.zeros_like(inputs[0], dtype=torch.float32)
        for x in inputs:
            acc += x.to(torch.float32)
        return acc.to(inputs[0].dtype)
    return torch.cat(inputs, dim=-1)


def _atol(op_name, dtype):
    """all_gather is data movement, so effectively exact. all_reduce sums world_size values, and
    bf16's 7-bit mantissa (ULP ~8x fp16's) makes tree-vs-sequential accumulation diverge by a few
    ULPs -- benign, but a CORRECT bf16 reduce needs a dtype-aware tolerance or it reads as a failure.
    A real bug is orders of magnitude past this."""
    if op_name == "all_gather":
        return 1e-3
    return 0.1 if dtype == torch.bfloat16 else 0.01


# The interface, all three impls and the `make_communicator` selector live in aiter's
# `communicator.py` -- exactly what the serving path runs, so this drives that selector directly.


# What each backend name MUST construct. Stated independently of the factory's if-chain on purpose:
# if the two ever disagree, one of them is wrong and this is what says so. A factory branch wired to
# the wrong class is silent -- you ask for iris, get something else, and the run produces a
# plausible number for the wrong thing.
#
# CONTROL FIRST in this dict, because the run order and the control derive from it -- one list of
# names, not two that can drift apart.
_BACKEND_CLASS = {
    "torch": TorchCommunicator,     # the known-good reference
    "hip": HipCommunicator,
    # "iris": IrisCommunicator,     # someone else's kernel; re-enable to measure against it
}

# Control FIRST, because it is the outermost pytest parameter and therefore the first case to run:
# if torch is red, nothing after it means anything.
BACKENDS = tuple(_BACKEND_CLASS)


# Enumerated rather than property-generated: a shrinking framework cannot drive across spawned ranks,
# and each candidate would be a full 8-process run.

DTYPES = ("fp16", "bf16")
# [tokens, 8192] is what vLLM hands a TP=8 all-reduce: hidden size 8192, token count varying with the
# batch. 4088 straddles nothing but is deliberately NOT a power of two, and 512 is the token count at
# which [tokens, 8192] bf16 reaches the 8 MiB admission bound -- so 511/512 sit either side of the
# edge where the communicator starts declining and vLLM falls back.
SHAPES = ((4, 8192), (128, 8192), (256, 8192), (511, 8192), (512, 8192), (4088, 8192))

# TWO modes, not three. `graph` ALWAYS varies the input across replays, because the
# identical-input variant could not catch the bug this suite exists for: if replay k+1 reads k's
# buffer before k's writes land, it gets k's data -- which EQUALS the correct answer when every
# input is identical, so the test passes while the race is live. vLLM copies a fresh activation in
# every token, so a stale read there is the previous token's garbage; varying is the faithful mode.
#
# Keeping identical-input as a third mode bought exactly one thing -- telling "capture itself is
# broken" apart from "staleness between replays" -- and `Measurement.worst_at` already gives that:
# diverging at replay 0 is capture, at replay 1+ is staleness. Same answer, half the matrix.
# THE THREE MODES, as data. Each is a SCHEDULE the one runner interprets, so the difference between
# them is this table rather than three code paths.
#
#   eager  does the collective compute the right answer AT ALL -- one call, no graph.
#   graph  does capture/replay PRESERVE that -- one collective, many replays, fresh input each time.
#   vllm   does the REAL pattern work -- a decode graph is a whole forward pass, so one capture holds
#          one collective PER LAYER on its own buffer. The aim is that passing here means running in
#          vLLM works; eager and graph exist to localise it when this one does not.
#
# `vllm` is a superset of `graph`: same replay depth, plus many buffers. It needs no separate
# eager-then-graph field, because a captured mode already warms up with eager calls on the same
# communicator before it captures -- which is what registers the staging buffer.
VLLM_LAYERS = 8

# The share of one card a single case may need for its inputs and snapshots. Multi-GPU runs OWN the
# machine (coordinated beforehand), so a case may take most of a card -- but not so much that the
# framework's own allocations turn a valid case into an OOM.
MEMORY_BUDGET = 0.70


@dataclass(frozen=True)
class Schedule:
    buffers: int        # distinct input buffers the collective is called on, per replay
    replays: int        # how many times the body runs
    captured: bool      # is the body recorded into a cudagraph and replayed

    @property
    def slots(self) -> int:
        """Total collectives, and the index space the inputs and the judging share."""
        return self.buffers * self.replays


# `shapes` differs because the modes ask different questions, not to save time. `graph` is the one
# that sweeps sizes, since capture/replay is where a size-dependent bug would live. `vllm` asks whether
# the PATTERN works -- one collective per layer inside one capture -- and one size answers that.
#
# It is also the difference between a 20-minute run and an overnight one. Each rank process builds
# EVERY rank's input for EVERY slot on the CPU (deliberately: CPU generation is bit-identical across
# processes regardless of device). At `vllm`'s 1600 slots over four shapes that is 4 TiB of `randn`
# across the node, which would dwarf the GPU work it exists to measure.
SCHEDULES = {
    "eager": Schedule(buffers=1, replays=1, captured=False),
    "graph": Schedule(buffers=1, replays=GRAPH_REPLAYS, captured=True),
    "vllm": Schedule(buffers=VLLM_LAYERS, replays=GRAPH_REPLAYS, captured=True),
}
MODES = tuple(SCHEDULES)


@dataclass(frozen=True)
class Measurement:
    """What a case that RAN produced. Pure numbers; no error channel."""

    # The ranks' own allclose verdict, STORED rather than re-derived. Deriving it as
    # `worst_diff <= atol` would be absolute-only, and a correct large-magnitude fp16 reduce exceeds
    # a fixed atol through rounding alone -- so the derived form would fail cases the ranks passed.
    within_tolerance: bool
    worst_diff: float
    worst_at: int               # replay index of the worst divergence; -1 for eager
    atol: float


def _build_communicator(backend, cpu_group, device_group, device):
    comm = make_communicator(cpu_group, device_group, device, backend=backend)
    # Every test funnels through here, so this one assertion covers the mapping at every
    # world size, dtype, shape and op the suite runs -- there is no separate test to
    # remember to extend when a backend is added.
    expected = _BACKEND_CLASS.get(backend)
    if expected is None:
        raise RuntimeError(f"test does not know what backend {backend!r} should build")
    if type(comm) is not expected:
        raise RuntimeError(
            f"asked for backend {backend!r} and got {type(comm).__name__}, "
            f"expected {expected.__name__} -- the factory is wired to the wrong class"
        )
    if comm.disabled:
        raise RuntimeError(f"{backend} communicator disabled")
    return comm


def _collect(pool, rets):
    """Every rank's result, or raise `mp.TimeoutError` once `CASE_TIMEOUT_S` is up.

    `pool.join()` cannot be used here: it waits forever, so a deadlocked rank stalls the whole run
    with nothing printed. `terminate()` frees the GPUs for the next case.
    """
    deadline = time.monotonic() + CASE_TIMEOUT_S
    out = []
    try:
        for r in rets:
            out.append(r.get(timeout=max(1.0, deadline - time.monotonic())))
    except mp.TimeoutError:
        pool.terminate()
        pool.join()
        raise
    pool.join()
    return out


INPUT_POOL = 16


def gen_inputs(shape, dtype, world: int, slots: int, device):
    """THE one source of input: every rank's tensor for every slot, as `[rank][slot]`.

    A cycled POOL of `INPUT_POOL` distinct tensors, not one per slot: what a replay must detect is a
    read of the PREVIOUS slot's data, so consecutive slots differing is what matters, not all 1600
    being unique. Per-slot interface, per-pool memory -- 1 GiB rather than 100 at `vllm`'s 1600 slots,
    which is what lets every shape run in every mode.
    """
    pool = min(slots, INPUT_POOL)
    made = [[_one_input(r, j, shape, dtype).to(device) for j in range(pool)] for r in range(world)]
    return [[made[r][j % pool] for j in range(slots)] for r in range(world)]


def _one_input(rank, k, shape, dtype):
    """Deterministic input for (rank, replay k), generated on CPU.

    CPU generation is bit-identical in every rank's process regardless of device (no reliance on
    cross-GPU randn determinism), which is what lets each rank rebuild the FULL per-replay reference
    locally -- including the other ranks' inputs -- without shipping tensors across the process
    boundary. Deterministic rather than random on purpose: a failure has to be re-runnable, and an
    exact reference makes any delta a real bug rather than fp noise.
    """
    g = torch.Generator().manual_seed(_INPUT_SEED + rank * 1_000_003 + k)
    return torch.randn(shape, generator=g).to(dtype)


def run_collective(comm, op_name, mine, sched: Schedule):
    """One output per SLOT, in slot order. Slot j is replay `j // buffers` on buffer `j % buffers`.

    The graph is a LOCAL and dies when this returns; a live one makes `destroy_process_group` block
    forever. The warmup before it runs EAGER on the same communicator, registering the staging buffer.
    Only a snapshot copy sits between replays, so they stay back-to-back -- an elided end barrier needs
    that to race.
    """
    statics = [mine[m].clone() for m in range(sched.buffers)]
    # `partial` on the API itself. Nothing re-checks admission: the communicator enforces its own
    # envelope and raises, so a check here would restate what the callee guarantees.
    ops = [partial(getattr(comm, op_name), st) for st in statics]

    def body():
        return [op() for op in ops]

    def feed(replay):
        for m, st in enumerate(statics):
            st.copy_(mine[replay * sched.buffers + m])

    if not sched.captured:
        out = []
        for k in range(sched.replays):
            feed(k)
            out += [o.clone() for o in body()]
        return out

    for _ in range(3):          # eager warmup: first-call allocations, and the staging registration
        body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(graph):
        outs = body()
    snaps = [torch.empty((sched.replays, *o.shape), dtype=o.dtype, device=o.device) for o in outs]
    for k in range(sched.replays):
        feed(k)
        graph.replay()
        for m, o in enumerate(outs):
            snaps[m][k].copy_(o)
    torch.cuda.synchronize()
    return [snaps[m][k] for k in range(sched.replays) for m in range(len(outs))]


def expected_outputs(op_name, inputs, slots: int):
    """What this rank should hold after each slot: the collective applied to every rank's input.

    POOLED like the inputs, and for the same reason -- only `INPUT_POOL` inputs are distinct, so only
    that many answers are. Building one per slot would have cost as much as the snapshots (100 GiB at
    `vllm`'s largest cell, on top of the snapshots' 100) and OOM'd inside the memory budget.
    """
    pool = min(slots, INPUT_POOL)
    made = [_expected(op_name, [inputs[r][j] for r in range(len(inputs))]) for j in range(pool)]
    return [made[j % pool] for j in range(slots)]


def compare(got, expected, atol):
    """Every replay against its OWN reference. Pure. Returns (ok, worst_diff, worst_at, atol).

    allclose semantics (atol + rtol*|ref|), not absolute-only: a correct large-magnitude reduce in
    fp16 exceeds a fixed 0.01 through rounding alone, and the torch control caught exactly that.
    `worst_at` is the diagnostic that let the identical-input mode be deleted -- diverging at replay
    0 means capture is wrong, at replay 1+ means a stale read between replays.
    """
    ok, worst_diff, worst_slot = True, 0.0, -1
    for j, (mine, ref) in enumerate(zip(got, expected)):
        a, b = mine.to(torch.float32), ref.to(torch.float32)
        d = (a - b).abs().max().item()
        if d > worst_diff:
            worst_diff, worst_slot = d, j
        if not torch.allclose(a, b, atol=atol, rtol=0.01):
            ok = False
    return ok, worst_diff, worst_slot


def exercise(backend: str, sched: Schedule, world: int, rank: int, device,
             cpu_group, group) -> tuple:
    """CREATE a communicator, exercise its whole API, TEAR IT DOWN. Returns the worst verdict.

    It owns the lifetime because construction and teardown are two of the ways a communicator fails --
    building one is a collective, and teardown releases IPC handles -- and owning both puts the order
    beyond reach. A declined cell, or one past the memory budget, is reported and skipped: declining is
    correct behaviour. Each cell is scoped so its tensors die with the frame.
    """
    budget = int(torch.cuda.get_device_properties(device).total_memory * MEMORY_BUDGET)
    worst = (True, 0.0, -1, 0.0)
    # `with`, so the release is in the SYNTAX: `Communicator.close()` runs at block exit whatever
    # happens inside, where `del` would only release if nothing else held a reference -- and a failing
    # cell's traceback holds the frames that hold the communicator, so `del` fails exactly when it
    # matters. The graph is already gone (a local of `_run`), and the caller destroys the process group
    # after this returns, so the three lifetimes nest correctly by construction.
    with _build_communicator(backend, cpu_group, group, device) as comm:
        for op_name, dtype_name, shape in product(OPS, DTYPES, SHAPES):
            dtype = dtypes.d_dtypes[dtype_name]
            one = torch.empty(shape, dtype=dtype)
            where = f"{op_name:11} {dtype_name:5} {str(shape):12}"

            if not getattr(comm, f"should_{op_name.replace('_', '')}")(one):
                _say(rank, f"      - {where} declined by the communicator")
                continue
            # Every rank materialises every rank's input for every slot, plus one snapshot each, and a
            # gather's output is world_size x its input. Derived from `_expected` so the estimate
            # cannot disagree with what the cell actually allocates.
            fan = _expected(op_name, [one] * world).numel() // one.numel()
            per = one.numel() * one.element_size()
            pool = min(sched.slots, INPUT_POOL)
            # All THREE things a cell holds. The snapshots are the only term that scales with slots;
            # inputs and expectations are pooled. Counting only two of the three is how this guard
            # passed a cell that then OOM'd.
            need = sched.slots * per * fan + pool * per * world + pool * per * fan
            if need > budget:
                _say(rank, f"      - {where} needs {need / 2**30:.0f}G, "
                           f"budget {budget / 2**30:.0f}G")
                continue

            atol = _atol(op_name, dtype)

            def cell():
                inputs = gen_inputs(shape, dtype, world, sched.slots, device)
                got = run_collective(comm, op_name, inputs[rank], sched)
                expected = expected_outputs(op_name, inputs, sched.slots)
                return compare(got, expected, atol)

            ok, diff, at = cell()
            _say(rank, f"      {where} worst|diff|={diff:g} atol={atol:g}"
                       + (f" @replay {at}" if at >= 0 else ""))
            worst = (worst[0] and ok, max(worst[1], diff),
                     at if diff > worst[1] else worst[2], atol)
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return worst


def run_rank(rank, world, pp, backend, mode, init_method):
    """ONE per-rank worker. It owns the PROCESS GROUP; the communicator's life is `exercise`'s.

    Every rank rebuilds every rank's input from a seed, so it judges its own replays and only scalars
    cross the process boundary.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(world_size=world, rank=rank,
                                 distributed_init_method=init_method)
    ensure_model_parallel_initialized(world, pp)
    cpu_group, group = get_tp_group().cpu_group, get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)     # force comm init before we measure
    torch.cuda.synchronize()
    # The communicator's whole life is `exercise`'s; this function owns only the PROCESS GROUP it
    # lives in. FINALLY, because plenty can go wrong in between and a rank that dies with its group
    # still up leaves its peers waiting on a socket instead of seeing a clean disconnect -- which
    # turns one rank's error into everyone's 600-second timeout.
    try:
        return exercise(backend, SCHEDULES[mode], world, rank, device, cpu_group, group)
    except BaseException:
        # Logged HERE, with the rank, before it crosses the process boundary: the pool surfaces one
        # failure to the parent, and on eight ranks the one it picks is not always the informative one.
        logger.exception("rank %d failed exercising %s/%s", rank, backend, mode)
        raise
    finally:
        # Its OWN try, so a teardown that fails cannot replace the failure that got us here -- the
        # diagnosis is worth more than the cleanup, and `destroy_process_group` is exactly the call
        # that has hung on us before.
        try:
            if dist.is_initialized():
                destroy_model_parallel()
                destroy_distributed_environment()
            torch.cuda.empty_cache()
        except BaseException:
            logger.exception("rank %d: teardown failed after %s/%s", rank, backend, mode)


def run_communicator(backend: str, mode: str, world: int, addr: str, port: int,
                     pp: int = 1) -> Measurement:
    """Exercise one backend in one mode across `world` ranks: spawn, collect under a timeout, take
    the worst rank's numbers.

    It RAISES. pytest is the boundary that turns a failure -- a timeout included -- into one red
    result. A collective's bug is often visible on only a subset of ranks, so the verdict is the
    WORST rank's and it passes only if every rank passed.
    """
    pool = Pool(processes=world)
    init = get_distributed_init_method(addr, port)
    try:
        rets = [pool.apply_async(run_rank, args=(r, world, pp, backend, mode, init))
                for r in range(world)]
        pool.close()
        per_rank = _collect(pool, rets)
    finally:
        pool.terminate()               # frees the GPUs whether it passed, failed or hung

    ok = all(r[0] for r in per_rank)
    _, worst_diff, worst_at, atol = max(per_rank, key=lambda r: r[1])
    return Measurement(within_tolerance=ok, worst_diff=worst_diff, worst_at=worst_at, atol=atol)


@pytest.fixture(scope="session")
def world() -> int:
    """Every GPU on the box. A read of the machine, so it is a fixture rather than a constant."""
    return torch.cuda.device_count()


@pytest.fixture
def rendezvous() -> Tuple[str, int]:
    """A FRESH port per case: the cases are sequential and each tears its group down, but two runs
    on a shared box must not collide."""
    return "127.0.0.1", get_open_port()


def baseline_admits(nbytes: int) -> bool:
    """`should_custom_ar` for a contiguous input on a fully-connected box, from the numbers above."""
    return nbytes % BASELINE_ALIGNMENT == 0 and nbytes < BASELINE_MAX_SIZE

@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_admission_matches_the_baseline(world_size: int) -> None:
    """Every backend admits exactly what vLLM's CustomAllreduce does.

    THE precondition for the matrix meaning anything: if an arm takes a tensor the baseline declines,
    the arms route different work and the numbers compare routing, not kernels.

    Needs no GPU, so it holds even where the matrix cannot run. fp32 is excluded and expected to
    diverge -- `hip_comms.cu` has fp16 and bf16 instantiations only, so closing that needs a kernel.
    """
    ours = object.__new__(TorchCommunicator)
    ours.disabled, ours.world_size, ours.max_size = False, world_size, BASELINE_MAX_SIZE
    # Every power of two across the range PLUS the bound and one element either side. Bounds alone
    # are the edges of the rule AS IT IS, so a wrong rule that diverges in the band between two of
    # them shows up on neither: a bounds-only grid missed a real bound at world 2 and 4.
    edges = tuple(1 << k for k in range(4, 28)) + (BASELINE_MAX_SIZE,)
    for dtype in (torch.float16, torch.bfloat16):
        es = torch.empty(0, dtype=dtype).element_size()
        for nbytes in sorted({e + d for e in edges for d in (-es, 0, es) if e + d > 0}):
            if nbytes % es:
                continue
            t = torch.empty(nbytes // es, dtype=dtype)
            want = baseline_admits(nbytes)
            assert ours.should_allreduce(t) == want, (
                f"all_reduce admission diverged from CustomAllreduce: {dtype} {nbytes}B "
                f"world={world_size} baseline={want}")
            assert ours.should_allgather(t) == want, (
                f"all_gather admission diverged: {dtype} {nbytes}B world={world_size}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_communicator(backend: str, mode: str, world: int, rendezvous: Tuple[str, int]) -> None:
    """Does this communicator work? One instance, its whole API, in one mode.

    Not parameterised per collective or per shape: `exercise` sweeps those on ONE communicator, which
    is what vLLM does, and it prints each cell's `worst|diff|` so a failure names the cell without
    needing 96 pytest ids. MODE is the ladder -- if `vllm` is red and `eager`/`graph` are green, the
    pattern is at fault rather than the arithmetic. BACKEND is outermost, so torch runs first: if the
    control is red, nothing after it means anything.
    """
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    addr, port = rendezvous
    print(f"\n  {backend} / {mode}", flush=True)
    got = run_communicator(backend, mode, world, addr, port)
    print(f"      => worst|diff|={got.worst_diff:g} atol={got.atol:g}", flush=True)
    assert got.within_tolerance, f"{backend}/{mode}: outside tolerance (cells above)"
