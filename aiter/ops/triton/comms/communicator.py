# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from typing import Iterator, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Match CustomAllreduce default (8 MB).
_DEFAULT_MAX_SIZE = 8 * 1024 * 1024


# The module the HIP backend's kernel comes from. Named ONCE: aiter's csrc carries no comms at all
# (it is GEMM/CK), so the kernel ships as its own extension the way iris does, and only this constant
# changes if that home changes.
_HIP_COMMS_MODULE = "aiter_hip_comms"


def _hip_comms_available() -> bool:
    import importlib.util

    return importlib.util.find_spec(_HIP_COMMS_MODULE) is not None


def _iris_available() -> bool:
    try:
        import iris  # noqa: F401

        return True
    except ImportError:
        return False


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _rocm_arch_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95"])
    except Exception:
        return False


class Communicator(ABC):
    """Interface for a TP all-reduce / all-gather backend behind vLLM's
    CudaCommunicator.

    Two implementations select at one factory (make_communicator):
    IrisCommunicator (production — iris gluon GPU-initiated CCL) and
    TorchCommunicator (a torch.distributed reference, the known-good control the
    iris path is measured and checked against). The surface mirrors what
    CudaCommunicator calls: a ``disabled`` flag, the should_*/all_* pairs
    (collectives are out-of-place — input untouched, new tensor returned), and a
    capture() context entered around cudagraph capture.
    """

    disabled: bool

    @abstractmethod
    def should_allreduce(self, inp: torch.Tensor) -> bool: ...

    @abstractmethod
    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def should_allgather(self, inp: torch.Tensor) -> bool: ...

    @abstractmethod
    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor: ...

    @abstractmethod
    def capture(self) -> AbstractContextManager[None]: ...


class IrisCommunicator(Communicator):
    """Communicator using Iris CCL GPU-initiated communication.

    API mirrors CustomAllreduce: __init__(cpu_group, device_group, device,
    max_size), should_allreduce, all_reduce (out-of-place), capture, plus
    disabled. Iris drives its own GPU-initiated CCL over a symmetric heap, so it
    uses neither torch group for collectives; it accepts both for interface
    parity with the other backends (and any future CPU-side coordination).
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**33  # 8 GB
    _AG_SLAB_SIZE = 2**25  # 32 MB per rank

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self.disabled = True
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._shmem = None
        self._workspace = None
        self._input_buf = None
        self._buf_shape = None
        self._buf_dtype = None
        self._ag_input_slab = None
        self._ag_output_slab = None

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        if not _rocm_arch_available():
            logger.debug("IrisCommunicator disabled: unsupported ROCm arch")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        try:
            import iris
            from iris.ccl.config import Config

            self._shmem = iris.iris(heap_size=self._HEAP_SIZE)
            self._gluon_config = Config(use_gluon=True)
        except Exception as e:
            logger.warning("Failed to initialize Allreduce: %s", e)
            return

        world_size = self._shmem.num_ranks
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug(
                "IrisCommunicator disabled: world_size=%d not in %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        self.disabled = False
        logger.info(
            "IrisCommunicator ready: world_size=%d heap=%dGB max_size=%dMB",
            world_size,
            self._HEAP_SIZE >> 30,
            self.max_size >> 20,
        )

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._shmem is None:
            return False
        if not _is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if inp_size * 2 > self._HEAP_SIZE:
            return False
        return True

    def _get_buffers(self, shape, dtype):
        if self._buf_shape != shape or self._buf_dtype != dtype:
            assert self._shmem is not None
            self._input_buf = self._shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None
        try:
            out = torch.empty_like(inp)
            input_buf = self._get_buffers(inp.shape, inp.dtype)
            input_buf.copy_(inp)

            if self._workspace is None:
                self._workspace = self._shmem.ccl.all_reduce_preamble(
                    out, input_buf, config=self._gluon_config
                )
            self._workspace = self._shmem.ccl.all_reduce(
                out,
                input_buf,
                workspace=self._workspace,
                config=self._gluon_config,
                async_op=True,
            )

            return out
        except Exception as e:
            logger.error(
                "IrisCommunicator.all_reduce failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    def should_allgather(self, inp: torch.Tensor) -> bool:
        """Replace every NCCL all_gather; only slab capacity falls back."""
        if self.disabled or self._shmem is None:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size > self._AG_SLAB_SIZE:
            logger.warning(
                "IrisCommunicator.all_gather fallback to NCCL: %d bytes "
                "exceeds slab",
                inp_size,
            )
            return False
        return True

    def _get_allgather_buffers(self, numel, dtype):
        # Fixed byte slabs allocated once; per-call views avoid heap churn
        # (the symmetric heap never frees).
        if self._ag_input_slab is None:
            assert self._shmem is not None
            world_size = self._shmem.num_ranks
            self._ag_input_slab = self._shmem.empty(
                (self._AG_SLAB_SIZE,), dtype=torch.uint8
            )
            self._ag_output_slab = self._shmem.empty(
                (world_size, self._AG_SLAB_SIZE), dtype=torch.uint8
            )
        input_buf = self._ag_input_slab.view(dtype)[:numel].view(1, numel)
        output_buf = self._ag_output_slab.view(dtype)[:, :numel]
        return input_buf, output_buf

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert self._shmem is not None
        try:
            if dim < 0:
                dim += inp.dim()
            world_size = self._shmem.num_ranks
            input_size = inp.size()

            input_buf, output_buf = self._get_allgather_buffers(inp.numel(), inp.dtype)
            input_buf.view(-1).copy_(inp.reshape(-1))

            self._shmem.ccl.all_gather(
                output_buf,
                input_buf,
                config=self._gluon_config,
                async_op=True,
            )

            # Same reshape contract as vLLM's DeviceCommunicatorBase.all_gather.
            # output_buf is a non-contiguous slab view, so reshape always
            # copies; the result never aliases the symmetric heap.
            output = output_buf.reshape((world_size,) + input_size).movedim(0, dim)
            return output.reshape(
                input_size[:dim]
                + (world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )

        except Exception as e:
            logger.error(
                "IrisCommunicator.all_gather failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    @contextmanager
    def capture(self) -> Iterator[None]:
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False


class TorchCommunicator(Communicator):
    """torch.distributed reference with the Communicator interface — the
    known-good control IrisCommunicator is measured and checked against.

    Same output contract as IrisCommunicator: all_reduce returns a new SUM tensor
    (input untouched); all_gather returns the per-rank inputs concatenated along
    ``dim``, rank-ordered. These are GPU-tensor collectives, so they run over the
    ``device_group`` (nccl/rccl); the gloo ``cpu_group`` (for CPU-object/IPC-handle
    handshakes) is accepted for interface parity but unused here.

    The should_* gates accept everything (torch.distributed is correct at any
    size/dtype), so this routes the same calls the iris path would plus the larger
    ones iris gates to NCCL. That is fine for the correctness control; aligning the
    gates with iris for routing parity is a perf-decomposition concern, not needed
    here.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.max_size = max_size
        self.world_size = dist.get_world_size(device_group)
        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def should_allgather(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.clone()
        dist.all_reduce(out, group=self.device_group)  # SUM
        return out

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += inp.dim()
        input_size = inp.size()
        out = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        dist.all_gather_into_tensor(out, inp.contiguous(), group=self.device_group)
        return out.movedim(0, dim).reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    @contextmanager
    def capture(self) -> Iterator[None]:
        # torch.distributed collectives need no special capture handling.
        yield


class DummyCommunicator(Communicator):
    """No-op communicator — the comms-free floor for reasoning about integration
    quality. Routes through the EXACT same path as iris (CudaCommunicator ->
    aiter_comm -> here) and preserves tensor shapes + cudagraph structure, but
    does ZERO cross-rank communication: all_reduce returns the input unreduced,
    all_gather replicates the local input. Output is therefore GARBAGE — this is
    perf-only, never eval it. Its value is the timing: dummy isolates the
    wrapper/dispatch path with no comm, so `iris - dummy` is iris's real
    comm+kernel+staging cost above that floor and `baseline - dummy` is baseline's
    comm cost. It uses no NCCL, so it captures cleanly at full cudagraph (unlike
    the torch backend, whose per-graph NCCL buffer registration OOMs)."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.max_size = max_size
        self.world_size = dist.get_world_size(device_group)
        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def should_allgather(self, inp: torch.Tensor) -> bool:
        return not self.disabled

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        # No-op: shape-preserving, no comm. Returns the (unreduced) input as a new
        # tensor to honor the out-of-place contract; the value is garbage.
        return inp.clone()

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += inp.dim()
        # Shape-correct no-op: replicate the local input across the world dim (no
        # comm). Matches the all_gather output shape; values are garbage.
        return torch.cat([inp] * self.world_size, dim=dim)

    @contextmanager
    def capture(self) -> Iterator[None]:
        yield


class HipCommunicator(Communicator):
    """Communicator backed by a HIP all-reduce kernel we own.

    The point of this backend is CONTROL, not (yet) speed. The iris backend's
    performance depends on someone else's kernel schedule; this one is ours, so it
    can be pushed as hard as the hardware allows without waiting on anyone. Its v1
    target is therefore PARITY with the incumbent, not beating it -- it is the
    platform every later comms experiment runs on.

    ALGORITHM (decided by measurement, 2026-07-30, not by preference): TWO-STAGE
    (reduce-scatter then all-gather), not one-shot. At the decode operating point
    a TP=8 all-reduce moves [64, 8192] bf16 = 1 MiB, and there a one-shot moves
    ~4x the bytes of a two-stage; measured, iris's `one_shot_all_reduce_gluon` ran
    ~37.8us against baseline's two-stage at ~22.2us on the same run. A one-shot
    only pays below ~1 MiB, so it is the wrong starting algorithm for this
    workload -- worth revisiting only if the message shrinks (lower concurrency or
    a smaller hidden dim).

    STATUS: the seam is LIVE, the kernel is pending. Without the kernel module the
    collectives run through torch -- correct values at torch speed -- so the backend
    is selectable, enabled, and exercised end to end while only the kernel remains
    to be written. It self-disables for the same reasons `IrisCommunicator` does
    (unsupported arch, unsupported world size), but NOT for a missing kernel.

    A placeholder that produces right answers is safe to run and dangerous to
    MEASURE, so it says so loudly at init, and `AITER_COMMS_REQUIRE_KERNEL=1` turns
    the fallback into a refusal for any run whose numbers are meant to mean
    something.
    """

    # Matches IrisCommunicator: a two-stage reduce-scatter needs the element count
    # divisible by the world size, and these are the TP widths we actually run.
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device_group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        # Disabled FIRST, so every early return below leaves a safe object rather
        # than one whose disabled flag depends on how far __init__ got.
        self.disabled = True
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.max_size = max_size
        self.world_size = dist.get_world_size(device_group)

        # A missing kernel selects the PLACEHOLDER; it does not disable the backend. Enabled is the
        # point: everything around the kernel -- the vLLM dispatch, the arm, the harness, the report
        # -- has to be exercised before the kernel exists, or the day the kernel lands is the day we
        # start debugging the plumbing instead of the kernel.
        self._kernel = _hip_comms_available()
        if not self._kernel:
            if os.environ.get("AITER_COMMS_REQUIRE_KERNEL") == "1":
                # For a run whose NUMBERS matter: refuse rather than quietly measure the placeholder.
                # A backend that silently substitutes something slower is a false READY -- the run
                # produces output and the output is about the wrong thing.
                raise RuntimeError(
                    f"AITER_COMMS_BACKEND=hip with AITER_COMMS_REQUIRE_KERNEL=1, but "
                    f"{_HIP_COMMS_MODULE} is not importable, so the HIP kernel is absent. Refusing "
                    f"to fall back: unset the variable to measure the placeholder deliberately."
                )
            logger.warning(
                "aiter HipCommunicator: PLACEHOLDER -- %s is not importable, so collectives run "
                "through torch (correct results, torch speed). Any timing from this arm is NOT the "
                "HIP kernel. Set AITER_COMMS_REQUIRE_KERNEL=1 to make this a refusal.",
                _HIP_COMMS_MODULE,
            )
        if not _rocm_arch_available():
            logger.info("aiter HipCommunicator disabled: unsupported ROCm arch")
            return
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.info(
                "aiter HipCommunicator disabled: world_size=%d not in %s",
                self.world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return
        self.disabled = False

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        # The SAME admission rules as IrisCommunicator, deliberately: two backends
        # that accept different tensors are not comparable, and the whole reason
        # this one exists is to be measured against that one.
        if self.disabled:
            return False
        if not _is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        # A two-stage reduce-scatter splits the buffer across ranks, so a count
        # that does not divide is a correctness hazard rather than a slow path.
        if inp.numel() % self.world_size != 0:
            return False
        return True

    def should_allgather(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not _is_weak_contiguous(inp):
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """The two-stage HIP all-reduce. Placeholder body until the kernel lands.

        The placeholder is CORRECT and slow rather than fast and wrong: `dist.all_reduce` gives the
        same values the kernel must give, so everything downstream -- vLLM's dispatch, the arm's
        output, eval accuracy, the report -- is exercised for real, and the only thing left to build
        is the kernel. Returning zeros would have made every number garbage and a plumbing bug
        indistinguishable from the stub.

        THE ONE LINE TO REPLACE: swap the `dist.all_reduce` below for the kernel call. Everything
        else about this backend is already what it will be."""
        if not self._kernel:
            out = inp.clone()
            dist.all_reduce(out, group=self.device_group)  # SUM
            return out
        raise NotImplementedError(
            f"{_HIP_COMMS_MODULE} is importable but HipCommunicator.all_reduce does not call it "
            f"yet -- wire the two-stage kernel here."
        )

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather. Placeholder body until the kernel lands; see `all_reduce`."""
        if not self._kernel:
            if dim < 0:
                dim += inp.dim()
            input_size = inp.size()
            out = torch.empty(
                (self.world_size,) + tuple(input_size),
                dtype=inp.dtype,
                device=inp.device,
            )
            dist.all_gather_into_tensor(out, inp.contiguous(), group=self.device_group)
            return out.movedim(0, dim).reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        raise NotImplementedError(
            f"{_HIP_COMMS_MODULE} is importable but HipCommunicator.all_gather does not call it yet."
        )

    @contextmanager
    def capture(self) -> Iterator[None]:
        # Nothing to register while there is no kernel. The two-stage design will
        # want its peer buffers registered here, which is why the hook exists now.
        yield


def make_communicator(
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    device: Union[int, str, torch.device],
    max_size: int = _DEFAULT_MAX_SIZE,
    backend: Optional[str] = None,
) -> Communicator:
    """Construct the TP collective backend at the one branching point.

    Takes both of vLLM's process groups (mirroring DeviceCommunicatorBase): the
    gloo ``cpu_group`` (CPU-object/IPC-handle handshakes) and the nccl/rccl
    ``device_group`` (GPU tensor collectives). Each backend uses what it needs —
    the torch reference runs its collectives over ``device_group``; iris uses
    neither (its own symmetric-heap CCL).

    'iris' is the gluon GPU-initiated CCL; 'hip' is our own HIP kernel (the backend
    we control, so its schedule is not someone else's); 'torch' is the torch.distributed
    reference/control; 'dummy' is a no-op comms-free floor (perf only — garbage
    output). The caller (vLLM) stays backend-agnostic and passes nothing; the
    backend is then resolved from ``AITER_COMMS_BACKEND``. There is
    NO default — if neither the ``backend`` argument (used by the tests) nor the
    env var is set, this raises, so the backend is always an explicit choice.

    Returns the communicator without raising on *unavailability* — the caller
    checks ``.disabled`` (IrisCommunicator self-disables on unsupported arch /
    missing iris / unsupported world size). A missing or unknown backend is a
    config error, not unavailability, and raises.
    """
    if backend is None:
        backend = os.environ.get("AITER_COMMS_BACKEND")
    if backend is None:
        raise ValueError(
            "AITER_COMMS_BACKEND is not set; specify the communicator backend "
            "explicitly ('iris', 'hip', 'torch', or 'dummy')"
        )
    backend = backend.lower()
    logger.info("aiter make_communicator: backend=%s", backend)
    if backend == "iris":
        return IrisCommunicator(cpu_group, device_group, device, max_size)
    if backend == "torch":
        return TorchCommunicator(cpu_group, device_group, device, max_size)
    if backend == "dummy":
        return DummyCommunicator(cpu_group, device_group, device, max_size)
    if backend == "hip":
        return HipCommunicator(cpu_group, device_group, device, max_size)
    raise ValueError(f"unknown communicator backend {backend!r}")
