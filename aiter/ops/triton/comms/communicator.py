# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Iterator, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from . import hip_comms

logger = logging.getLogger(__name__)

# vLLM's CustomAllreduce default, and the only bound its admission uses. That class is what these
# backends replace, so the envelope is transcribed from it rather than invented.
_DEFAULT_MAX_SIZE = 8 * 1024 * 1024


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
    """A TP all-reduce / all-gather backend behind vLLM's CudaCommunicator.

    This class owns the call SEQUENCE -- the admission gate, the capture invariant, and delegation --
    and a backend supplies only the hooks at the bottom. Collectives are out-of-place.
    """

    disabled: bool
    max_size: int
    world_size: int

    # The admission envelope, shared by EVERY backend including torch. Uniform on purpose: torch is
    # the control, so a control that admits a superset is comparing against a different question --
    # a shape outside the envelope would run on torch and fall back on the others.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    # A class attribute, so a backend needs no cooperating `__init__` to get the invariant.
    _capturing: bool = False

    # Checked when the class is DEFINED, the earliest moment there is.
    _CALLERS_SURFACE = ("should_allreduce", "should_allgather", "all_reduce", "all_gather", "capture",
                        "_admits")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        taken = [n for n in Communicator._CALLERS_SURFACE if n in cls.__dict__]
        if taken:
            raise TypeError(
                f"{cls.__name__} overrides {taken}, which `Communicator` owns -- an override skips "
                f"the capture invariant and the admission gate. Supply `_all_reduce`, `_all_gather`, "
                f"or `_on_capture` instead -- admission is not a backend's to redefine."
            )

    # ---- What the CALLER uses. Concrete: this class owns the order. ----

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether this backend will take `inp`. Public because a False means the caller falls back."""
        return not self.disabled and self._admits(inp)

    def should_allgather(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._admits(inp)

    def _admits(self, inp: torch.Tensor) -> bool:
        """THE envelope, identical for every backend and both ops.

        Transcribed from `vllm.distributed.device_communicators.custom_all_reduce.should_custom_ar`,
        the path these backends replace: a 16-byte multiple, weak-contiguous, under `max_size`.
        Admitting a different set would change which tensors take the fast path. DTYPE is the one
        addition -- our kernels are instantiated for fp16 and bf16 only.

        The bound is the INPUT's, which is what our buffers hold: hip stages the input in a `max_size`
        buffer, and iris's per-rank gather slab is larger still.
        """
        nbytes = inp.numel() * inp.element_size()
        return (
            _is_weak_contiguous(inp)
            and nbytes % 16 == 0
            and nbytes < self.max_size
            and inp.dtype in self._SUPPORTED_DTYPES
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        self._check_capture("all_reduce")
        if not self.should_allreduce(inp):
            raise RuntimeError(self._rejected("all_reduce", inp))
        return self._all_reduce(inp)

    def all_gather(self, inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._check_capture("all_gather")
        if not self.should_allgather(inp):
            raise RuntimeError(self._rejected("all_gather", inp))
        return self._all_gather(inp, dim)

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Enter around a cudagraph capture. Required: a captured launch records an address that is
        not valid yet, so a backend has to be told."""
        self._capturing = True
        try:
            with self._on_capture():
                yield
        finally:
            self._capturing = False

    # ---- The two rules a caller can get wrong, enforced once. ----

    def _check_capture(self, op: str) -> None:
        """Refuse a collective recorded into a graph outside `capture()`: the stream says it is
        capturing and this object says nobody entered the context, so the caller is wrong."""
        if torch.cuda.is_current_stream_capturing() and not self._capturing:
            raise RuntimeError(
                f"{type(self).__name__}.{op} is being captured into a cudagraph without "
                f"`capture()`. Use `with comm.capture(), torch.cuda.graph(g): ...` -- a backend may "
                f"defer peer registration until that context exits, and a graph captured without it "
                f"replays against addresses that were never registered."
            )

    def _rejected(self, op: str, inp: torch.Tensor) -> str:
        return (f"{type(self).__name__} rejected {op}: shape={tuple(inp.shape)} dtype={inp.dtype} "
                f"disabled={self.disabled}. Ask should_{op.replace('_', '')} first and fall back "
                f"when it says no.")

    # ---- What a BACKEND supplies. ----

    @abstractmethod
    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """SUM across ranks, out of place: input untouched, new tensor returned. Assume `inp` is
        admitted -- the base checked."""

    @abstractmethod
    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """The per-rank inputs concatenated along `dim`, rank-ordered."""

    def _on_capture(self) -> AbstractContextManager[None]:
        """What this backend needs around a capture. Nothing, by default."""
        return nullcontext()


class IrisCommunicator(Communicator):
    """Communicator using Iris CCL GPU-initiated communication.

    API mirrors CustomAllreduce: __init__(cpu_group, device_group, device,
    max_size), should_allreduce, all_reduce (out-of-place), capture, plus
    disabled. Iris drives its own GPU-initiated CCL over a symmetric heap, so it
    uses neither torch group for collectives; it accepts both for interface
    parity with the other backends (and any future CPU-side coordination).
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
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
        self.world_size = world_size
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug(
                "IrisCommunicator disabled: world_size=%d not in %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        # The heap and the slab are limits on the CONFIGURATION, not on a tensor: if `max_size` fits
        # inside both, no admitted input can exceed either, so nothing needs re-checking per call.
        if max_size * 2 > self._HEAP_SIZE or max_size > self._AG_SLAB_SIZE:
            logger.warning(
                "IrisCommunicator disabled: heap=%dGB / slab=%dMB cannot back the admitted bounds",
                self._HEAP_SIZE >> 30, self._AG_SLAB_SIZE >> 20,
            )
            return
        self.disabled = False
        logger.info(
            "IrisCommunicator ready: world_size=%d heap=%dGB max_size=%dMB",
            world_size,
            self._HEAP_SIZE >> 30,
            self.max_size >> 20,
        )

    # No admission of its own: `_shmem is None` already means `disabled`, and the heap and slab are
    # checked against `max_size` at construction.

    def _get_buffers(self, shape, dtype):
        if self._buf_shape != shape or self._buf_dtype != dtype:
            assert self._shmem is not None
            self._input_buf = self._shmem.empty(shape, dtype=dtype)
            self._buf_shape = shape
            self._buf_dtype = dtype
            self._workspace = None
        return self._input_buf

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
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

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
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

    # No `_on_capture`: iris needs nothing around a capture.


class TorchCommunicator(Communicator):
    """torch.distributed reference: the known-good control the other backends are checked against.

    Collectives run over `device_group` (nccl/rccl); the gloo `cpu_group` is accepted for interface
    parity and unused. It admits exactly what the others admit, taking the shared envelope unchanged.
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

    # Supplies neither `_admits_*` nor `_on_capture`: it takes the shared envelope unchanged -- the
    # control has to admit exactly what it is a control for -- and needs no capture handling.

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.clone()
        dist.all_reduce(out, group=self.device_group)  # SUM
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
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



class HipCommunicator(Communicator):
    """Communicator over HIP collectives we own: `hip_comms.cu` beside this file, compiled by
    `hip_comms.py`, depending on torch and the HIP runtime only.

    Self-disables on an unsupported arch or world size. A failed COMPILE instead RAISES -- disabling
    would let vLLM fall back to its own all-reduce and report the run READY.
    """

    # Matches IrisCommunicator: a two-stage reduce-scatter needs the element count
    # divisible by the world size, and these are the TP widths we actually run.
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

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

        # EAGER, and after the disable checks: compiling inside vLLM's cudagraph capture is
        # not recoverable, and a box that cannot run this backend should not pay a build.
        # The context owns the peer handshake and is built once per group, like
        # CustomAllreduce; it knows nothing about which collective runs over it.
        #
        # NOTE this line is a COLLECTIVE (it all-gathers IPC handles), so every rank must
        # reach it. The disable checks above are uniform across a TP group in practice --
        # same arch, same world size -- but if they ever were not, the ranks that got here
        # would HANG waiting for the ones that returned, rather than failing. Worth
        # knowing because a deadlock is far worse than an error.
        # The staging buffer backs the EAGER path and holds the largest input we admit.
        self._comms = hip_comms.HipComms(cpu_group, self.device, max_size=self.max_size)
        self.disabled = False
        logger.info(
            "HipCommunicator ready: world_size=%d max_size=%dMB",
            self.world_size,
            self.max_size >> 20,
        )

    # No admission of its own. A two-stage reduce-scatter will need the count to divide the ranks;
    # the shipped kernel is one-shot and does not, and the baseline does not check it either, so
    # adding it would refuse tensors both we and the path we replace can handle.


    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Sum `inp` across the TP ranks. The launch config is chosen in `hip_comms`."""
        out = torch.empty_like(inp)
        self._comms.all_reduce(out, inp)
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered."""
        return self._comms.all_gather(inp.contiguous(), dim)

    def _on_capture(self) -> AbstractContextManager[None]:
        # A captured input's address is not registered when the launch is recorded, so the context
        # reserves a slot during capture and exchanges the IPC handles on the way out.
        return self._comms.capture()


def make_communicator(
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    device: Union[int, str, torch.device],
    max_size: int = _DEFAULT_MAX_SIZE,
    backend: Optional[str] = None,
) -> Communicator:
    """Construct the TP collective backend at the one branching point.

    Takes both of vLLM's process groups and each backend uses what it needs. The backend comes from
    the `backend` argument or `AITER_COMMS_BACKEND`, with NO default, so it is always an explicit
    choice; a missing or unknown one raises.

    Unavailability does NOT raise -- the caller checks `.disabled`. A config error does.
    """
    if backend is None:
        backend = os.environ.get("AITER_COMMS_BACKEND")
    if backend is None:
        raise ValueError(
            "AITER_COMMS_BACKEND is not set; specify the communicator backend "
            "explicitly ('iris', 'hip', or 'torch')"
        )
    backend = backend.lower()
    logger.info("aiter make_communicator: backend=%s", backend)
    if backend == "iris":
        return IrisCommunicator(cpu_group, device_group, device, max_size)
    if backend == "torch":
        return TorchCommunicator(cpu_group, device_group, device, max_size)
    if backend == "hip":
        return HipCommunicator(cpu_group, device_group, device, max_size)
    raise ValueError(f"unknown communicator backend {backend!r}")
