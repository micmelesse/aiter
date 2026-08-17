# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Python interface to our HIP collectives, and the only place they are TUNED.

Two files, both ours: this and `hip_comms.cu`. `torch.utils.cpp_extension.load` compiles
the `.cu` with hipcc and caches the `.so`, so there is no build system, no entry in
aiter's `optCompilerConfig.json`, and nothing in aiter's `csrc`.

THE SPLIT. The `.cu` is mechanism and holds no policy; every decision -- which algorithm,
how many blocks, how many threads -- is made here, by `config_for`, and passed down. Same
shape as a Triton kernel with `@triton.autotune`: the kernel body has no heuristics and
the meta-parameters are chosen outside it. Two consequences worth knowing:

- A tuning decision is a pure Python function, so it is testable without a GPU. A `if
  (size < N)` in the `.cu` would be a decision nobody can see, and the env var it
  eventually grows is how you end up with `VLLM_CUSTOM_ALLREDUCE_ALGO`.
- It costs nothing where it matters. vLLM captures cudagraphs, so `config_for` runs ONCE
  during capture and replay is pure kernel launch with no Python at all.

`ngpus` and the dtype have to be compile-time to unroll and vectorize, so they select a
template instantiation rather than being passed; the `.cu`'s dispatch names every
combination that exists and REFUSES anything else rather than substituting.

Two environment knobs, both set by the arm's Dockerfile rather than defaulted here:
`TORCH_EXTENSIONS_DIR` (where the `.so` is cached; must be an image path, not the mounted
`~/.cache`) and `PYTORCH_ROCM_ARCH` (which torch turns into `--offload-arch`; required
when warming the build with no GPU present to detect).
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# The extension's import name. Only used for the build cache and error messages: the
# module object is held here, never imported by name from anywhere else.
NAME = "aiter_hip_comms"

SOURCE = Path(__file__).resolve().parent / "hip_comms.cu"

# Algorithms, matching the `.cu`'s dispatch. One today.
ALGO_ONE_SHOT = 0

_module: Any = None
_lock = threading.Lock()


def load(verbose: bool = False) -> Any:
    """Compile (first call) and return the extension. Idempotent; raises on failure.

    Raising is deliberate. A missing iris is genuine unavailability and self-disables,
    but our own source failing to build is a broken toolchain or broken code, and
    disabling would let vLLM fall back to its own all-reduce and call the run READY.
    """
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        if not SOURCE.is_file():
            raise RuntimeError(f"{NAME}: source is missing at {SOURCE}")
        from torch.utils.cpp_extension import load as _cpp_load

        _module = _cpp_load(name=NAME, sources=[str(SOURCE)], verbose=verbose)
        return _module


# =================================================================================
# TUNING. The whole tuning surface is this dataclass and the function under it.
# =================================================================================


@dataclass(frozen=True)
class LaunchConfig:
    """How to run one collective. Pure data, chosen in Python, passed to the kernel."""

    algo: int = ALGO_ONE_SHOT
    blocks: int = 16
    threads: int = 512


# A single default, deliberately. Every field here is a guess until we have a measurement,
# and a tuning table invented before the first number is a wrong abstraction held
# confidently. `blocks=16` is vLLM's tuned value on the same hardware, carried over on the
# grounds that a measured constant beats an unmeasured one -- their comment is that too
# many SMs contend on the interconnect. When we have numbers this becomes a lookup keyed
# by (op, dtype, numel, world_size); nothing outside this function needs to change.
_DEFAULT = LaunchConfig()


def config_for(op: str, dtype: torch.dtype, numel: int, world_size: int) -> LaunchConfig:
    """The tuning decision, and the ONLY place one is made."""
    return _DEFAULT


# =================================================================================
# CONTEXT. Peer memory and registration, one per process group.
# =================================================================================


def _all_gather_object(group: ProcessGroup, obj: Any) -> List[Any]:
    out: List[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


class HipComms:
    """The peer-memory context: IPC handshake, signal block, scratch, registration.

    ONE per process group, like vLLM's `CustomAllreduce`, because peer pointers are
    group-scoped. It knows nothing about which collective runs over it, which is the
    property that keeps a new workload from touching it.

    The handle EXCHANGE happens here rather than in C++: it is a collective over the
    gloo `cpu_group`, and a process group is not something the kernel layer should know
    about. C++ only opens the handles it is handed.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        *,
        scratch_bytes: int = 8 << 20,
        max_buffers: int = 512,
    ) -> None:
        mod = load()
        self.mod = mod
        self.cpu_group = cpu_group
        self.device = device
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)

        # ONE allocation per rank holds the signal block and the scratch after it, so the
        # two-stage algorithm needs no new buffer or handshake -- only a kernel.
        self._signal = torch.zeros(
            mod.SIGNAL_BYTES + scratch_bytes, dtype=torch.uint8, device=device
        )
        # Device-side array of peer-pointer sets, one slot per registered buffer.
        self._slab = torch.zeros(
            mod.PEER_PTRS_BYTES * max_buffers, dtype=torch.uint8, device=device
        )

        handles, offsets = self._exchange(self._signal.data_ptr())
        self.comms = mod.Comms(
            rank=self.rank,
            world_size=self.world_size,
            self_signal=self._signal.data_ptr(),
            signal_handles=handles,
            signal_offsets=offsets,
            peer_slab=self._slab.data_ptr(),
            peer_slab_bytes=self._slab.numel(),
        )

    def _exchange(self, ptr: int) -> Tuple[List[bytes], List[int]]:
        """Every rank's IPC handle + offset for its own `ptr`, in rank order."""
        mine = self.mod.ipc_handle_and_offset(ptr)
        gathered = _all_gather_object(self.cpu_group, mine)
        return [h for h, _ in gathered], [o for _, o in gathered]

    def register(self, tensor: torch.Tensor) -> None:
        """Make `tensor` usable as a collective INPUT. Collective: every rank must call it
        for its own tensor, in the same order."""
        handles, offsets = self._exchange(tensor.data_ptr())
        self.comms.register_buffer(
            handles=handles, offsets=offsets, self_ptr=tensor.data_ptr()
        )

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Wrap a cudagraph capture. Buffers used inside are registered on EXIT.

        While capturing, an input's address is not registered yet, so the kernel layer
        reserves a slot and records the pointer; here we exchange handles for everything
        recorded and fill those slots in. Sound because a captured address is fixed for
        the graph's life -- the replayed kernel reads a peer-pointer set populated after
        the capture that recorded its launch.
        """
        yield
        self.flush_pending()

    def flush_pending(self) -> None:
        """Register whatever the capture deferred. A no-op when nothing is pending."""
        pending: Sequence[int] = self.comms.pending_graph_buffers()
        # Collective: every rank must agree on the count, or the exchange below deadlocks
        # with a confusing message instead of this one.
        counts = _all_gather_object(self.cpu_group, len(pending))
        if len(set(counts)) != 1:
            raise RuntimeError(
                f"hip_comms: ranks captured different numbers of buffers ({counts}); "
                "every rank must run the same graph."
            )
        if not pending:
            return
        per_buffer = [self._exchange(p) for p in pending]
        self.comms.register_graph_buffers(
            handles=[h for h, _ in per_buffer], offsets=[o for _, o in per_buffer]
        )

    def all_reduce(
        self, out: torch.Tensor, inp: torch.Tensor, cfg: Optional[LaunchConfig] = None
    ) -> None:
        """Sum `inp` across every rank into `out`, in place."""
        if cfg is None:
            cfg = config_for("all_reduce", inp.dtype, inp.numel(), self.world_size)
        self.comms.all_reduce(
            out=out, inp=inp, algo=cfg.algo, blocks=cfg.blocks, threads=cfg.threads
        )

    def all_gather(
        self, inp: torch.Tensor, dim: int = -1, cfg: Optional[LaunchConfig] = None
    ) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered.

        The kernel fills a RANK-MAJOR `(world, *inp.shape)` buffer and the axis is moved
        here, which is what the torch path did -- so the two agree bit for bit and the
        reference in the correctness suite covers both. The `movedim`+`reshape` copy is a
        known cost and a later perf item, not a correctness one.
        """
        if cfg is None:
            cfg = config_for("all_gather", inp.dtype, inp.numel(), self.world_size)
        if dim < 0:
            dim += inp.dim()
        shape = tuple(inp.size())
        staged = torch.empty((self.world_size,) + shape, dtype=inp.dtype, device=inp.device)
        self.comms.all_gather(
            out=staged, inp=inp, algo=cfg.algo, blocks=cfg.blocks, threads=cfg.threads
        )
        return staged.movedim(0, dim).reshape(
            shape[:dim] + (self.world_size * shape[dim],) + shape[dim + 1 :]
        )
