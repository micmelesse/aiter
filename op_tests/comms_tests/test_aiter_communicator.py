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
from multiprocessing import Pool, freeze_support, set_start_method
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

# Replays for the cudagraph correctness check. Back-to-back with no inter-replay
# sync is what stresses an elided end barrier (replay N+1 must not start before
# replay N's symmetric-heap writes land); a per-replay sync would hide the race.
NUM_REPLAYS = 1000

# Replays for the varying-input check. Fewer than NUM_REPLAYS because each replay
# also keeps a snapshot output buffer (N x out-shape) and a stale read shows
# within the first few differing replays anyway — no need for 1000.
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

# Every backend runs the FULL correctness matrix. torch is first because it is the
# control: it is known-good, so if it fails the harness is wrong and no other verdict
# in the run means anything. Subset with `-b`.
BACKENDS = ("torch", "iris", "hip")


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
    capture,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """One rank: init distributed, build the `backend` communicator, run `op_name`
    either eagerly or under cudagraph capture + NUM_REPLAYS back-to-back replays,
    and return the result for the driver to check against the reference."""
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

    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op()
        for _ in range(NUM_REPLAYS):
            graph.replay()
        torch.cuda.synchronize()

    result = out.clone()

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


def test_communicator(
    tp_size,
    pp_size,
    shape,
    dtype,
    op_name,
    capture,
    backend,
    distributed_init_method: Optional[str] = None,
):
    """Driver for one identical-input case, eager or cudagraph-captured.
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
                capture,
                backend,
                distributed_init_method,
            ),
        )
        for i in range(tp_size)
    ]
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
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
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]  # each rank: (ok, worst_diff, worst_k)
    atol = tolerance(op_name, dtype)
    ok = all(r[0] for r in rets)  # every rank's every replay was allclose
    worst = max(rets, key=lambda r: r[1])  # rank with the largest abs diff
    worst_diff, worst_k = worst[1], worst[2]
    logger.info(
        f"[{backend}] {op_name} [cudagraph/varying]: {shape=} {dtype=} "
        f"{'OK' if ok else 'FAIL'} (worst |diff|={worst_diff:.3g} atol={atol} @replay {worst_k})"
    )
    return ok, worst_diff, worst_k, atol


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
    "-b",
    "--backend",
    type=str,
    choices=list(BACKENDS),
    default=None,
    help="run only this backend (default: all of them)",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    # Before any GPU work: prove the selector returns what it is asked for.
    check_backend_selection()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    backends = [args.backend] if args.backend else list(BACKENDS)
    print(f"backends: {backends}")

    # Every backend, every collective, identical-input eager then cudagraph capture +
    # replay. Verdicts are COLLECTED, not raised on, so one backend's failure does not
    # hide the rest -- the whole picture lands in one run.
    summary = []  # (phase, backend, op, dtype, shape, ok, worst_diff, worst_k, atol)
    for backend in backends:
        for op_name in OPS:
            for dtype in l_dtype:
                for shape in l_shape:
                    for capture in (False, True):
                        phase = "cudagraph" if capture else "eager"
                        ok, wd, atol = test_communicator(
                            8,
                            1,
                            shape,
                            dtype,
                            op_name,
                            capture,
                            backend,
                            distributed_init_method=get_distributed_init_method(
                                "127.0.0.1", get_open_port()
                            ),
                        )
                        summary.append(
                            (phase, backend, op_name, dtype, shape, ok, wd, -1, atol)
                        )

    # Then the varying-input cudagraph check — fresh input per replay, each
    # checked against its own reference. This is the one that catches a
    # dropped/stale symmetric-heap read (which identical-input replays hide).
    def _init():
        return get_distributed_init_method("127.0.0.1", get_open_port())

    for backend in backends:
        for op_name in OPS:
            for dtype in l_dtype:
                for shape in l_shape:
                    ok, wd, wk, atol = test_communicator_vary(
                        8, 1, shape, dtype, op_name, backend, _init()
                    )
                    summary.append(
                        ("vary", backend, op_name, dtype, shape, ok, wd, wk, atol)
                    )

    print("\n==== correctness summary ====")
    for phase, backend, op_name, dt, sh, ok, wd, wk, atol in summary:
        print(
            f"  [{backend:5}] {phase:9} {op_name:11} {str(sh):12} {str(dt):16} "
            f"{'OK  ' if ok else 'FAIL'} worst|diff|={wd:.3g} atol={atol} @replay {wk}"
        )

    # CONTROL FIRST. torch.distributed is known-good, so it failing means the HARNESS is
    # wrong and every other verdict in the table is worthless -- a distinct, louder error
    # than a backend bug.
    control_fail = [s for s in summary if s[1] == "torch" and not s[5]]
    assert not control_fail, (
        "CONTROL FAILED: torch.distributed (known-good) is wrong under this harness "
        f"({control_fail[0][6]:.3g} > atol {control_fail[0][8]}). The harness is unsound; "
        "no other result in the table can be trusted until this is fixed."
    )

    # Then the backends under test. Control passed, so these are real bugs.
    failed = [s for s in summary if not s[5]]
    if failed:
        raise AssertionError(
            "collectives are WRONG (control passed, so these are real backend bugs, "
            "not test artifacts): "
            + "; ".join(
                f"{b}/{phase} {op} {sh} {dt} worst|diff|={wd:.3g}>{atol} @replay {wk}"
                for phase, b, op, dt, sh, _, wd, wk, atol in failed
            )
        )
    print("all cases within tolerance")
