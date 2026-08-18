# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of EVERY communicator backend's collective ops (all_reduce and
all_gather) in eager mode, under cudagraph capture + replay, AND under capture +
replay with the input changing every replay.

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

import argparse
import logging
import os
from dataclasses import dataclass
import multiprocessing as mp
import time
from multiprocessing import Pool, freeze_support, set_start_method
from typing import List, Sequence, Tuple
from typing_extensions import Optional

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
    IrisCommunicator,
    TorchCommunicator,
    make_communicator,
)

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

# Replays for the cudagraph case. Back-to-back with no inter-replay sync is what stresses an
# elided end barrier (replay N+1 must not start before replay N's writes land); a per-replay sync
# would hide the race. 200 rather than more because each replay also keeps a snapshot output buffer
# (N x out-shape) and a stale read shows within the first few differing replays anyway.
NUM_VARY_REPLAYS = 200

# Deterministic per-(rank, replay) seed base for the varying-input check.
_VARY_SEED = 20260615

OPS = ["all_reduce", "all_gather"]


# ── Communicator: one interface, three impls, one branching point ──
# The interface (Communicator ABC), all three impls -- IrisCommunicator, HipCommunicator
# (ours) and TorchCommunicator (the known-good control) -- and the make_communicator
# selector all live in aiter's communicator.py, which is exactly what the serving
# path runs. This test drives that selector directly: every backend runs the same
# matrix, and "torch" is the control because it is the known-good one.
# make_communicator returns the communicator without raising on unavailability, so
# _build_communicator checks `.disabled` here.


# What each backend name MUST construct. Stated here independently of the factory's
# if-chain on purpose: if the two ever disagree, one of them is wrong and this is what
# says so. A factory branch wired to the wrong class is silent -- you ask for iris, get
# something else, and the run produces a plausible number for the wrong thing.
_BACKEND_CLASS = {
    "iris": IrisCommunicator,
    "hip": HipCommunicator,
    "torch": TorchCommunicator,
}

# Every backend runs the FULL correctness matrix, and the ORDER is deliberate. torch is
# first because it is the control: known-good, so if it fails the harness is wrong and no
# other verdict in the run means anything. `hip` is second because it is the one under
# active development -- the full matrix takes hours, and putting the backend we are
# iterating on last means waiting most of that before learning anything about it.
# Subset with `-b`.
BACKENDS = ("torch", "hip", "iris")


# ── THE DOMAIN: one type, one generator ──
# What this suite explores is a TYPE, not four module-level lists read in a five-deep nest. A
# reader answers "what does this cover" from `Case` and `cases()` alone, and one failing case is
# nameable, so re-running exactly it is possible.
#
# NOT `@given`: our default is property-first (CONTRIBUTING *Property tests by default*), and this
# is the stated distributed exception -- Hypothesis cannot drive across spawned ranks and each
# shrink step would be a full 8-process run, so the sweep is hand-coded (CONTRIBUTING *Shrink and
# bisect the input axis*). The domain is small and enumerated on purpose.

# TWO modes, not three. `cudagraph` ALWAYS varies the input across replays, because the
# identical-input variant could not catch the bug this suite exists for: if replay k+1 reads k's
# buffer before k's writes land, it gets k's data -- which EQUALS the correct answer when every
# input is identical, so the test passes while the race is live. vLLM copies a fresh activation in
# every token, so a stale read there is the previous token's garbage; varying is the faithful mode.
#
# Keeping identical-input as a third mode bought exactly one thing -- telling "capture itself is
# broken" apart from "staleness between replays" -- and `Measurement.worst_at` already gives that:
# diverging at replay 0 is capture, at replay 1+ is staleness. Same answer, half the matrix.
MODES = ("eager", "cudagraph")


@dataclass(frozen=True)
class Case:
    """ONE unit of work: which backend, which collective, at what dtype and shape, replayed how.

    `mode` is the axis that used to be duplicated code -- `eager`/`cudagraph` went through one
    driver and `varying` through a near-identical second one.
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

    worst_diff: float
    worst_at: int               # replay index of the worst divergence; -1 for eager
    atol: float

    @property
    def within_tolerance(self) -> bool:
        return self.worst_diff <= self.atol


@dataclass(frozen=True)
class Outcome:
    """What became of one case: it measured something, or it failed with an exception.

    ONE optional-pair rather than `ok`/`hung`/`error` flags. Those could contradict each other
    (`ok=True, hung=True` was representable and meaningless) and flattened the exception to a
    string, losing the traceback. `hung` is not a field at all now -- it is
    `isinstance(failure, mp.TimeoutError)`, which is where that fact actually lives.
    """

    case: Case
    measured: Optional[Measurement] = None
    failure: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.measured is not None and self.measured.within_tolerance

    @property
    def hung(self) -> bool:
        return isinstance(self.failure, mp.TimeoutError)

    @property
    def label(self) -> str:
        if self.hung:
            return "HUNG"
        if self.failure is not None:
            return "ERROR"
        return "OK" if self.ok else "FAIL"


def cases(backends: Sequence[str], ops: Sequence[str], dts: Sequence["torch.dtype"],
          shapes: Sequence[Tuple[int, ...]], modes: Sequence[str] = MODES) -> List[Case]:
    """THE matrix, and the only place its order is decided.

    BACKEND is outermost so the control runs first: torch is known-good, so if it fails the harness
    is unsound and no later verdict means anything. MODE is innermost so everything about one
    (backend, op, dtype, shape) is known before moving on -- with the modes split across two phases,
    a hang in `varying` was reachable only after the whole identical-input matrix had passed.
    """
    return [Case(b, op, dt, sh, m)
            for b in backends for op in ops for dt in dts for sh in shapes for m in modes]


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


def check_backend_selection():
    """The selector's contract, with no GPU and no process group: every known name maps
    to its own class, an unknown name RAISES, and there is no default.

    Cheap and first, because the failure it catches is the expensive kind -- a silently
    wrong backend does not crash, it returns numbers for something you did not ask for.
    """
    for name, cls in _BACKEND_CLASS.items():
        assert name in _BACKEND_CLASS and cls is not None
    # distinct classes: a copy-paste in the factory that points two names at one impl
    # would make two "different" arms the same measurement.
    assert len(set(_BACKEND_CLASS.values())) == len(_BACKEND_CLASS)

    for bad in ("hpi", "Iris ", "", "nccl"):
        try:
            make_communicator(None, None, 0, backend=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"backend {bad!r} was accepted; it must raise")

    saved = os.environ.pop("AITER_COMMS_BACKEND", None)
    try:
        make_communicator(None, None, 0, backend=None)
    except ValueError:
        pass
    else:
        raise AssertionError("an unset backend must raise, never pick a default")
    finally:
        if saved is not None:
            os.environ["AITER_COMMS_BACKEND"] = saved

    logging.info("backend selection OK: %s", ", ".join(sorted(_BACKEND_CLASS)))


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


def run_comm(
    tp_size,
    pp_size,
    rankID,
    x,
    op_name,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """One rank of an EAGER case: init distributed, build the `backend` communicator, call
    `op_name` once, and return the result for the driver to check against the reference.

    Eager only. Capture lives in `run_comm_vary`, because a captured graph is only worth replaying
    against a CHANGING input -- replaying an identical one cannot catch a stale read (see MODES)."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    cpu_group = get_tp_group().cpu_group
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    comm = _build_communicator(backend, cpu_group, group, device)
    op = _make_op(comm, op_name, x)

    # Warm up eagerly so first-call allocations (workspace, symmetric buffers)
    # happen before capture — graph capture can't perform them cleanly.
    for _ in range(3):
        out = op()
    torch.cuda.synchronize()

    result = out.clone()

    # FREE THE GRAPH BEFORE TEARING DOWN THE PROCESS GROUP. A captured graph holds references to
    # the NCCL communicator's work and buffers, and `destroy_process_group` blocks draining work a
    # live graph still owns -- every rank then waits inside it forever. Seen 2026-08-18: all 8 ranks
    # stuck in `destroy_process_group` on the `all_gather` capture case, GPUs at 0%, for 8 minutes
    # until it was killed. `out` goes too: it is a graph-pool tensor, so it keeps the pool alive.
    del out
    torch.cuda.synchronize()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return result


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


CASE_TIMEOUT_S = 600


def _collect(pool, rets):
    """Every rank's result, or raise `mp.TimeoutError` once `CASE_TIMEOUT_S` is up.

    THE reason this exists rather than `pool.join()`: join waits forever. On 2026-08-18 all eight
    ranks deadlocked inside `destroy_process_group` and the driver sat in join with nothing printed
    for eight minutes. `AsyncResult.get(timeout=...)` is what makes a hang a reportable outcome, and
    `terminate()` is what stops the ranks so the next case can have the GPUs.
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


def test_communicator(
    tp_size,
    pp_size,
    shape,
    dtype,
    op_name,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """Driver for one EAGER case.
    Returns (ok, worst_diff, atol)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    inputs = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]
    ref = reference(op_name, inputs)
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            run_comm,
            args=(
                tp_size,
                pp_size,
                i,
                inputs[i],
                op_name,
                backend,
                distributed_init_method,
            ),
        )
        for i in range(tp_size)
    ]
    pool.close()
    rets = _collect(pool, rets)
    # Return a verdict rather than raising, for the same reason the varying-input
    # driver does: the caller runs EVERY backend before deciding, so one backend's
    # failure must not abort the others. Same allclose semantics as that path.
    atol = tolerance(op_name, dtype)
    rtol = 0.01
    ref32 = ref.to(torch.float32)
    ok = True
    worst_diff = 0.0
    for out in rets:
        got = out.to(ref).to(torch.float32)
        worst_diff = max(worst_diff, (got - ref32).abs().max().item())
        if not torch.allclose(got, ref32, atol=atol, rtol=rtol):
            ok = False
    return ok, worst_diff, atol


def _vary_input(rank, k, shape, dtype):
    """Deterministic input for (rank, replay k), generated on CPU.

    CPU generation is bit-identical across every rank's process regardless of
    device (no reliance on cross-GPU randn determinism), which is what lets each
    rank rebuild the full per-replay reference locally — including the OTHER
    ranks' inputs — without shipping data around. The caller moves it to the
    device. A wrong reference would mean false failures, so this is generated the
    bulletproof way even though it costs some CPU."""
    g = torch.Generator().manual_seed(_VARY_SEED + rank * 1_000_003 + k)
    return torch.randn(shape, generator=g).to(dtype)


def run_comm_vary(
    tp_size,
    pp_size,
    rankID,
    shape,
    dtype,
    op_name,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """One rank of the varying-input cudagraph check, for `backend` (any of BACKENDS;
    'torch' is the known-good control).

    Captures the op once, then replays NUM_VARY_REPLAYS times, copying a DIFFERENT
    input into the static capture buffer before each replay — exactly how vLLM
    drives a captured decode step (new activations written into the static input
    every token). Only a cheap snapshot copy sits between replays so they stay
    back-to-back (an elided end barrier needs that to race); all checking happens
    after a single sync. Each replay's output is compared to its own reference;
    returns (worst_abs_diff, worst_replay_index) for the driver."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    cpu_group = get_tp_group().cpu_group
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    comm = _build_communicator(backend, cpu_group, group, device)

    # Full deterministic input matrix on this device: every rank's input for
    # every replay. We need all ranks' inputs (not just ours) to build the
    # per-replay reference after the run.
    all_inputs = [
        [_vary_input(r, k, shape, dtype).to(device) for k in range(NUM_VARY_REPLAYS)]
        for r in range(tp_size)
    ]
    my_inputs = all_inputs[rankID]

    static_in = my_inputs[0].clone()
    op = _make_op(comm, op_name, static_in)

    # Warm up eagerly so first-call allocations happen before capture.
    for _ in range(3):
        out = op()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op()

    # Back-to-back replays: fresh input in, snapshot out, no inter-replay sync.
    out_buf = torch.empty(
        (NUM_VARY_REPLAYS, *out.shape), dtype=out.dtype, device=device
    )
    for k in range(NUM_VARY_REPLAYS):
        static_in.copy_(my_inputs[k])
        graph.replay()
        out_buf[k].copy_(out)  # snapshot before the next replay overwrites `out`
    torch.cuda.synchronize()

    # Check each replay against its own reference (post-sync; host side is fine).
    # allclose semantics (atol + rtol*|ref|), SAME as the eager/identical path —
    # absolute-only would flag a correct large-magnitude reduce (fp16 rounding of
    # a sum of 8 values exceeds a fixed 0.01 even though the relative error is
    # tiny; the torch control caught exactly that). worst_diff is reported for
    # context; `ok` is the allclose verdict.
    atol = tolerance(op_name, dtype)
    rtol = 0.01
    ok = True
    worst_diff = 0.0
    worst_k = -1
    for k in range(NUM_VARY_REPLAYS):
        ref_k = reference(op_name, [all_inputs[r][k] for r in range(tp_size)]).to(
            torch.float32
        )
        got = out_buf[k].to(torch.float32)
        d = (got - ref_k).abs().max().item()
        if d > worst_diff:
            worst_diff, worst_k = d, k
        if not torch.allclose(got, ref_k, atol=atol, rtol=rtol):
            ok = False

    # Same teardown deadlock as `run_comm`: free the graph (and its pool tensors) before the
    # process group, or every rank hangs in `destroy_process_group` draining work the graph owns.
    del graph, out, out_buf
    torch.cuda.synchronize()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return ok, worst_diff, worst_k


def test_communicator_vary(
    tp_size,
    pp_size,
    shape,
    dtype,
    op_name,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """Driver for one varying-input cudagraph case. Each rank self-checks every
    replay against the per-replay reference; we take the worst over ranks and
    return a verdict (the caller aggregates and decides pass/fail, so the control
    and iris arms both run before any assertion). A dropped/stale symmetric-heap
    read surfaces here as an O(1) diff — the failure mode the identical-input
    replay loop cannot see. Returns (ok, worst_diff, worst_k, atol)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            run_comm_vary,
            args=(
                tp_size,
                pp_size,
                i,
                shape,
                dtype,
                op_name,
                backend,
                distributed_init_method,
            ),
        )
        for i in range(tp_size)
    ]
    rets = _collect(pool, rets)      # each rank: (ok, worst_diff, worst_k)
    atol = tolerance(op_name, dtype)
    ok = all(r[0] for r in rets)  # every rank's every replay was allclose
    worst = max(rets, key=lambda r: r[1])  # rank with the largest abs diff
    worst_diff, worst_k = worst[1], worst[2]
    logger.info(
        f"[{backend}] {op_name} [cudagraph/varying]: {shape=} {dtype=} "
        f"{'OK' if ok else 'FAIL'} (worst |diff|={worst_diff:.3g} atol={atol} @replay {worst_k})"
    )
    return ok, worst_diff, worst_k, atol


# ── ONE driver, and the timeout that makes a hang REPORTABLE ──

# A case that has not answered in this long is hung, not slow. The whole matrix at its largest
# shape ran in well under a minute per case when it worked, so this is generous by an order of
# magnitude and still catches the failure it exists for. Without it a deadlocked case sits until
# gk's 4h workload deadline and the run reports nothing at all -- which is what happened on
# 2026-08-17 (killed at 30min by gloo's own timeout) and again on 2026-08-18 (killed by hand at 8min).
def run_case(case: Case, world: int = 8, pp: int = 1) -> Outcome:
    """Run ONE case across `world` ranks. THE boundary: the only place a case failure is caught.

    It catches rather than propagates because a per-case failure must not abandon the cases after
    it -- the summary is the deliverable, and one backend's bug must not hide another's. Everything
    below this raises normally (the pythonic default); this is the one seam that turns an exception
    into a recorded outcome, including a TIMEOUT, so a deadlocked case is reported instead of
    stalling the run silently.
    """
    init = get_distributed_init_method("127.0.0.1", get_open_port())
    try:
        if case.mode == "cudagraph":
            ok, worst, at, atol = test_communicator_vary(
                world, pp, case.shape, case.dtype, case.op, case.backend, init)
        else:
            ok, worst, atol = test_communicator(
                world, pp, case.shape, case.dtype, case.op, case.backend, init)
            at = -1
        return Outcome(case=case, measured=Measurement(worst_diff=worst, worst_at=at, atol=atol))
    except BaseException as exc:       # noqa: BLE001 -- a case failure is data, not a crash
        return Outcome(case=case, failure=exc)


def _detail(o: Outcome) -> str:
    """What one outcome actually knows. A hung case has no diff, and printing `worst|diff|=0` for
    one would read as a passing measurement."""
    if o.hung:
        return f"no answer in {CASE_TIMEOUT_S}s"
    if o.failure is not None:
        return f"{type(o.failure).__name__}: {o.failure}"
    m = o.measured
    assert m is not None                      # no failure and no measurement is unrepresentable
    at = f" @replay {m.worst_at}" if m.worst_at >= 0 else ""
    return f"worst|diff|={m.worst_diff:.3g} atol={m.atol}{at}"


def render(outcomes: Sequence[Outcome]) -> str:
    """The summary, as the deliverable: one line per case, in the order they ran. PURE."""
    return "\n".join(["", "==== correctness summary ===="]
                     + [f"  [{o.label:5}] {o.case}  {_detail(o)}" for o in outcomes])


def verdict_of_run(outcomes: Sequence[Outcome], control: str = "torch") -> Optional[str]:
    """The run's single conclusion, or None if everything passed. PURE.

    CONTROL FIRST, and separately: `torch` is known-good, so it failing means the HARNESS is
    unsound and every other line in the table is worthless. That is a different message from a
    backend bug, and conflating the two once cost a day of chasing the wrong thing.
    """
    bad = [o for o in outcomes if not o.ok]
    ctrl = [o for o in bad if o.case.backend == control]
    if ctrl:
        return (f"CONTROL FAILED: {control} is wrong or hung under this harness "
                f"({ctrl[0].case} -- {_detail(ctrl[0])}). The harness is unsound; no other "
                f"verdict in the table can be trusted.")
    if bad:
        hung = [o for o in bad if o.hung]
        lead = f"{len(hung)} case(s) HUNG; " if hung else ""
        return (f"{lead}{len(bad)} of {len(outcomes)} case(s) failed (control passed, so these are "
                f"real backend bugs): "
                + "; ".join(f"{o.case} [{o.label}]" for o in bad))
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Every input EXPLICIT, in one place: parse, build the matrix, run it, report, decide.

    The five steps are the whole program and they are visible in order. This replaced a 107-line
    `__main__` block that inlined two five-deep loop nests, its own progress counters, its own
    summary formatting and its own verdict logic -- so the matrix, the running and the judging
    could not be read or changed independently.
    """
    freeze_support()
    args = parser.parse_args(argv)

    # Cheap and first: no GPU, no process group. A silently wrong backend does not crash, it
    # returns numbers for something you did not ask for.
    check_backend_selection()

    backends = _pick(args.backend, BACKENDS, "backend")
    modes = _pick(args.mode, MODES, "mode")
    dts = ([dtypes.d_dtypes[args.dtype]] if args.dtype
           else [dtypes.d_dtypes[k] for k in l_dtype])
    shapes = [args.shape] if args.shape else list(l_shape)

    plan = cases(backends, OPS, dts, shapes, modes)
    print(f"world={args.world}  backends={backends}  modes={list(modes)}  "
          f"dtypes={[str(d) for d in dts]}  shapes={shapes}")
    print(f"{len(plan)} case(s), timeout {CASE_TIMEOUT_S}s each", flush=True)

    if args.list:
        for i, case in enumerate(plan, 1):
            print(f"  {i:3}  {case}")
        return 0

    outcomes: List[Outcome] = []
    for i, case in enumerate(plan, 1):
        # Announce BEFORE running: a hang leaves this line as the last thing printed, which names
        # the case that hung. Flushed, because a hung process never drains a buffer.
        print(f"[{i}/{len(plan)}] {case}", flush=True)
        o = run_case(case, world=args.world)
        print(f"      -> {o.label:5}  {_detail(o)}", flush=True)
        outcomes.append(o)

    print(render(outcomes), flush=True)
    problem = verdict_of_run(outcomes)
    if problem:
        print(f"\n{problem}", flush=True)
        return 1
    print(f"\nall {len(outcomes)} case(s) within tolerance", flush=True)
    return 0


def _pick(raw: Optional[str], known: Sequence[str], what: str) -> List[str]:
    """A comma-separated subset of `known`, in `known`'s order. Unknown names REFUSE rather than
    silently narrowing the run to nothing."""
    if not raw:
        return list(known)
    asked = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in asked if s not in known]
    if unknown:
        raise SystemExit(f"unknown {what}(s) {unknown}; known: {list(known)}")
    return [k for k in known if k in asked]


l_dtype = ["fp16", "bf16"]
l_shape = [(4, 8192), (128, 8192), (256, 8192)]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 128,8192",
)
parser.add_argument(
    "-l",
    "--list",
    action="store_true",
    help="print the matrix and exit, running nothing (what WILL run, before spending GPUs on it)",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default=None,
    help=f"comma-separated subset of {','.join(MODES)} (default: all)",
)
parser.add_argument(
    "-w",
    "--world",
    type=int,
    default=8,
    help="ranks to run each case across (default: 8)",
)
parser.add_argument(
    "-b",
    "--backend",
    type=str,
    default=None,
    help=f"comma-separated subset of {','.join(BACKENDS)} (default: all, in that order)",
)


if __name__ == "__main__":
    raise SystemExit(main())
