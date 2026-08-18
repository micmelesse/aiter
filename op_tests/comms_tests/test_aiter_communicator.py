# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of EVERY communicator backend's collective ops (all_reduce and
all_gather) in two modes: `eager` (one call) and `graph` (captured once, then replayed with a
FRESH input every replay).

One question per backend: does it produce correct results? Eager
alone is not enough — the gluon kernels elide barriers under graph capture, and a
race there only shows across a sequence of replays (vLLM captures the decode step
once and replays it every token).

But replaying the SAME input every time is also not enough: a stale-heap read
(replay k+1 reading replay k's symmetric-buffer before k's writes land) returns
the previous replay's data, and when every replay's input is identical the stale
data EQUALS the correct data, so the bug hides. vLLM never does that — it copies
a fresh activation into the static input buffer before each token's replay, so a
stale read there is the *previous token's* data = garbage. The varying-input
check (run_comm_vary) reproduces exactly that: fresh input per replay, each
replay's output checked against its own reference. That is the mode that catches
the dropped/stale all_gather race the identical-input loop misses.
"""

import logging
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
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.ops.triton.comms.communicator import (
    _DEFAULT_MAX_SIZE,
    HipCommunicator,
    IrisCommunicator,
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

OPS = ["all_reduce", "all_gather"]


# ── Communicator: one interface, three impls, one branching point ──
# The interface (Communicator ABC), all three impls -- IrisCommunicator, HipCommunicator
# (ours) and TorchCommunicator (the known-good control) -- and the make_communicator
# selector all live in aiter's communicator.py, which is exactly what the serving
# path runs. This test drives that selector directly: every backend runs the same
# matrix, and "torch" is the control because it is the known-good one.
# make_communicator returns the communicator without raising on unavailability, so
# _build_communicator checks `.disabled` here.


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
    "iris": IrisCommunicator,
}

# Control FIRST, because it is the outermost pytest parameter and therefore the first case to run:
# if torch is red, nothing after it means anything.
BACKENDS = tuple(_BACKEND_CLASS)


# ── THE DOMAIN: one type, one generator ──
# What this suite explores is a TYPE, not four module-level lists read in a five-deep nest. A
# reader answers "what does this cover" from `Case` and `cases()` alone, and one failing case is
# nameable, so re-running exactly it is possible.
#
# Enumerated rather than property-generated: a shrinking framework cannot drive across spawned
# ranks, and each candidate would be a full 8-process run.

# TWO modes, not three. `graph` ALWAYS varies the input across replays, because the
# identical-input variant could not catch the bug this suite exists for: if replay k+1 reads k's
# buffer before k's writes land, it gets k's data -- which EQUALS the correct answer when every
# input is identical, so the test passes while the race is live. vLLM copies a fresh activation in
# every token, so a stale read there is the previous token's garbage; varying is the faithful mode.
#
# Keeping identical-input as a third mode bought exactly one thing -- telling "capture itself is
# broken" apart from "staleness between replays" -- and `Measurement.worst_at` already gives that:
# diverging at replay 0 is capture, at replay 1+ is staleness. Same answer, half the matrix.
MODES = ("eager", "graph")


DTYPES = ("fp16", "bf16")
SHAPES = ((4, 8192), (128, 8192), (256, 8192))


@dataclass(frozen=True)
class Case:
    """ONE unit of work: which backend, which collective, at what dtype and shape, replayed how.

    """

    backend: str
    op: str
    dtype: "torch.dtype"
    shape: Tuple[int, ...]
    mode: str

    def __str__(self) -> str:
        return f"{self.backend:5} {self.op:11} {str(self.shape):12} {str(self.dtype):14} {self.mode}"


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


def _make_op(comm, op_name, x):
    """The collective under test as a zero-arg closure over the rank's input,
    after enforcing the communicator's own should_* precondition."""
    if op_name == "all_reduce":
        if not comm.should_allreduce(x):
            raise RuntimeError(
                f"{type(comm).__name__} rejected all_reduce: "
                f"shape={tuple(x.shape)} dtype={x.dtype}"
            )
        return lambda: comm.all_reduce(x)
    if op_name == "all_gather":
        if not comm.should_allgather(x):
            raise RuntimeError(
                f"{type(comm).__name__} rejected all_gather: "
                f"shape={tuple(x.shape)} dtype={x.dtype}"
            )
        return lambda: comm.all_gather(x)
    raise ValueError(f"unknown op {op_name!r}")


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


def _replay_input(rank, k, shape, dtype):
    """Deterministic input for (rank, replay k), generated on CPU.

    CPU generation is bit-identical in every rank's process regardless of device (no reliance on
    cross-GPU randn determinism), which is what lets each rank rebuild the FULL per-replay reference
    locally -- including the other ranks' inputs -- without shipping tensors across the process
    boundary. Deterministic rather than random on purpose: a failure has to be re-runnable, and an
    exact reference makes any delta a real bug rather than fp noise.
    """
    g = torch.Generator().manual_seed(_INPUT_SEED + rank * 1_000_003 + k)
    return torch.randn(shape, generator=g).to(dtype)


def _run_eager(comm, op, inputs, static_in):
    """Call the collective once. Returns one output per input, so eager is the 1-replay case.

    `comm` is unused, and taken anyway so both runners have ONE signature: the mode is chosen by
    assigning a runner, and two shapes would make that assignment carry a per-mode argument list."""
    static_in.copy_(inputs[0])
    return [op().clone()]


def _run_graph(comm, op, inputs, static_in):
    """Capture once, then replay with a FRESH input each time. One output per replay.

    Three things this shape is load-bearing for: `comm.capture()` is required (a backend may defer
    registration until it exits); the graph is a LOCAL so it dies before `destroy_process_group`,
    which otherwise blocks forever draining work a live graph owns; and only a snapshot copy sits
    between replays, because an elided end barrier needs them back-to-back to race.
    """
    for _ in range(3):          # warm up so first-call allocations happen before capture
        out = op()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(graph):
        out = op()
    snaps = torch.empty((len(inputs), *out.shape), dtype=out.dtype, device=out.device)
    for k, x in enumerate(inputs):
        static_in.copy_(x)
        graph.replay()
        snaps[k].copy_(out)     # snapshot before the next replay overwrites `out`
    torch.cuda.synchronize()
    return [snaps[k] for k in range(len(inputs))]


def reference(op_name, inputs, dim=-1):
    """What every rank should hold afterwards. all_reduce = elementwise sum
    (accumulated in fp32 so the reference itself doesn't eat bf16 rounding);
    all_gather = concat of the per-rank inputs along `dim`, rank-ordered (the
    Communicator.all_gather contract, same for every backend)."""
    if op_name == "all_reduce":
        acc = torch.zeros_like(inputs[0], dtype=torch.float32)
        for x in inputs:
            acc += x.to(torch.float32)
        return acc.to(inputs[0].dtype)
    if op_name == "all_gather":
        return torch.cat(inputs, dim=dim)
    raise ValueError(f"unknown op {op_name!r}")


def tolerance(op_name, dtype):
    """all_gather is pure data movement → effectively exact. all_reduce sums
    world_size values, and bf16's 7-bit mantissa (ULP ~8x fp16's) makes tree-vs-
    sequential accumulation diverge by a few ULPs — benign, but it needs a
    dtype-aware absolute tolerance so a *correct* bf16 reduce isn't flagged. A
    real reduction bug produces garbage orders of magnitude beyond this."""
    if op_name == "all_gather":
        return 1e-3
    return 0.1 if dtype == torch.bfloat16 else 0.01


def _judge(op_name, dtype, all_inputs, got):
    """Every replay against its OWN reference. Pure. Returns (ok, worst_diff, worst_at, atol).

    allclose semantics (atol + rtol*|ref|), not absolute-only: a correct large-magnitude reduce in
    fp16 exceeds a fixed 0.01 through rounding alone, and the torch control caught exactly that.
    `worst_at` is the diagnostic that let the identical-input mode be deleted -- diverging at replay
    0 means capture is wrong, at replay 1+ means a stale read between replays.
    """
    atol, rtol = tolerance(op_name, dtype), 0.01
    ok, worst_diff, worst_at = True, 0.0, -1
    for k, mine in enumerate(got):
        ref = reference(op_name, [all_inputs[r][k] for r in range(len(all_inputs))]).to(torch.float32)
        cur = mine.to(torch.float32)
        d = (cur - ref).abs().max().item()
        if d > worst_diff:
            worst_diff, worst_at = d, k
        if not torch.allclose(cur, ref, atol=atol, rtol=rtol):
            ok = False
    return ok, worst_diff, worst_at, atol


def run_rank(rank, world, pp, case, init_method):
    """ONE per-rank worker for EVERY case. Bring up, run the mode, judge, tear down.

    Every rank rebuilds every rank's input from a seed, so it can compute the reference itself and
    judge its own replays. Only scalars cross the process boundary.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(world_size=world, rank=rank,
                                 distributed_init_method=init_method)
    ensure_model_parallel_initialized(world, pp)
    cpu_group = get_tp_group().cpu_group
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)     # force comm init before we measure
    torch.cuda.synchronize()
    comm = _build_communicator(case.backend, cpu_group, group, device)

    replays = GRAPH_REPLAYS if case.mode == "graph" else 1
    all_inputs = [[_replay_input(r, k, case.shape, case.dtype).to(device)
                   for k in range(replays)] for r in range(world)]
    mine = all_inputs[rank]
    static_in = mine[0].clone()
    op = _make_op(comm, case.op, static_in)

    runner = _run_graph if case.mode == "graph" else _run_eager
    verdict = _judge(case.op, case.dtype, all_inputs, runner(comm, op, mine, static_in))

    # The ONE teardown in the file. Anything holding graph-pool memory goes first; the graph itself
    # is already freed by `_run_graph` having returned.
    del all_inputs, mine, static_in, op, comm
    torch.cuda.synchronize()
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return verdict


def run_case(case: Case, world: int, addr: str, port: int, pp: int = 1) -> Measurement:
    """Run ONE case across `world` ranks: spawn, collect under a timeout, take the worst rank's.

    It RAISES: one case per test makes pytest the boundary that turns a failure -- a timeout
    included -- into one red result rather than the end of the run.
    """
    # The rendezvous, and the ONLY channel for it. Deliberately not `MASTER_ADDR`/`MASTER_PORT`:
    # nothing in `aiter.dist` reads those, and torch consults them only for `init_method="env://"`.
    # A free port per case is what keeps two runs on a shared box from colliding.
    pool = Pool(processes=world)
    init = get_distributed_init_method(addr, port)
    try:
        rets = [pool.apply_async(run_rank, args=(r, world, pp, case, init)) for r in range(world)]
        pool.close()
        per_rank = _collect(pool, rets)
    finally:
        pool.terminate()               # frees the GPUs whether the case passed, failed or hung

    # EVERY rank's verdict, reduced to the worst. A collective's bug is often visible on only a
    # subset of ranks (a distance/topology effect), so one rank's view is a single data point --
    # the case passes only if every rank passed.
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


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_admission_matches_the_baseline(world_size: int) -> None:
    """Every backend admits exactly what `CustomAllreduce` does, which is the path they replace.

    THE precondition for the matrix below meaning anything: if an arm takes a tensor the baseline
    declines, the arms route different work and the numbers compare routing, not kernels.

    Needs no GPU -- both predicates read only tensor metadata plus a few attributes -- so it holds
    even where the matrix cannot run. fp32 is excluded and expected to diverge: `hip_comms.cu` has
    fp16 and bf16 instantiations only, so closing that needs a kernel, not a predicate.
    """
    def _as(cls, **kw):
        obj = object.__new__(cls)
        for k, v in kw.items():
            setattr(obj, k, v)
        return obj

    base = _as(CustomAllreduce, disabled=False, world_size=world_size, fully_connected=True,
               max_size=_DEFAULT_MAX_SIZE)
    ours = _as(TorchCommunicator, disabled=False, world_size=world_size, max_size=_DEFAULT_MAX_SIZE)
    # Every power of two across the range PLUS the current bounds and one element either side.
    # Bounds alone are the edges of the rules AS THEY ARE, so a wrong rule that diverges in the band
    # between two of them shows up on neither -- measured: the bound we shipped went unseen at
    # world 2 and 4 under a bounds-only grid.
    edges = tuple(1 << k for k in range(4, 28)) + (
        _DEFAULT_MAX_SIZE // (world_size * 2), _DEFAULT_MAX_SIZE, 8192 * 8192)
    for dtype in (torch.float16, torch.bfloat16):
        es = torch.empty(0, dtype=dtype).element_size()
        for nbytes in sorted({e + d for e in edges for d in (-es, 0, es) if e + d > 0}):
            if nbytes % es:
                continue
            t = torch.empty(nbytes // es, dtype=dtype)
            assert base.should_custom_ar(t) == ours.should_allreduce(t), (
                f"all_reduce admission diverged: {dtype} {nbytes}B world={world_size}")
            assert base.should_custom_ag(t) == ours.should_allgather(t), (
                f"all_gather admission diverged: {dtype} {nbytes}B world={world_size}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_collective_matches_the_reference(backend: str, op: str, dtype: str,
                                          shape: Tuple[int, ...], mode: str,
                                          world: int, rendezvous: Tuple[str, int]) -> None:
    """ONE case: one backend's one collective, at one dtype and shape, in one mode.

    A test PER CASE, so a hang or a fault is one red result rather than the end of the run, and
    `-k` selects instead of a flag. BACKEND is the outermost parameter, so the control runs first:
    if torch is red, nothing after it means anything.
    """
    if world < 2:
        pytest.skip("a collective needs at least two ranks")
    addr, port = rendezvous
    case = Case(backend=backend, op=op, dtype=dtypes.d_dtypes[dtype], shape=shape, mode=mode)
    got = run_case(case, world=world, addr=addr, port=port)
    assert got.within_tolerance, (
        f"{case}: worst|diff|={got.worst_diff:g} atol={got.atol:g}"
        + (f" @replay {got.worst_at}" if got.worst_at is not None else ""))
