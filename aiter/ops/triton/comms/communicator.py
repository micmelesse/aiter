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

# Match CustomAllreduce default (8 MB).
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
    """Interface for a TP all-reduce / all-gather backend behind vLLM's
    CudaCommunicator.

    Three implementations select at one factory (make_communicator): IrisCommunicator
    (iris gluon GPU-initiated CCL), HipCommunicator (our own HIP kernel) and
    TorchCommunicator (a torch.distributed reference, the known-good control the
    other two are measured and checked against). The surface mirrors what
    CudaCommunicator calls: a ``disabled`` flag, the should_*/all_* pairs
    (collectives are out-of-place — input untouched, new tensor returned), and a
    capture() context entered around cudagraph capture.

    THE SEQUENCE OF A CALL IS DECIDED HERE, once, and a backend supplies only the pieces it is
    asked for. It used to be five abstract methods, so every rule about HOW to call a collective was
    a rule the caller had to remember, and each one failed in a different register: skipping
    `capture()` was silent for torch, dead for iris and a GPU memory fault for hip; calling
    `all_reduce` without asking `should_allreduce` was checked by nobody at all. Now the caller
    cannot skip a step, because the step is not theirs to take. (LOG 2026-08-18.)
    """

    disabled: bool

    # A CLASS attribute, so a backend needs no cooperating `__init__` to get the invariant.
    _capturing: bool = False

    # The sequence is not a suggestion: a backend that re-decides it silently gets its own rules back,
    # which is the state this class was built to leave. Checked when the class is DEFINED -- the
    # earliest moment available, before any run, any GPU, and any test.
    _CALLERS_SURFACE = ("should_allreduce", "should_allgather", "all_reduce", "all_gather", "capture")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        taken = [n for n in Communicator._CALLERS_SURFACE if n in cls.__dict__]
        if taken:
            raise TypeError(
                f"{cls.__name__} overrides {taken}, which `Communicator` owns. Supply the pieces "
                f"instead: `_all_reduce`, `_all_gather`, `_admits_all_reduce`, `_admits_all_gather`, "
                f"`_on_capture`. The base enforces the capture invariant and the admission gate for "
                f"every backend, and an override skips both."
            )

    # ---- What the CALLER uses. Concrete: this class owns the order. ----

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Whether this backend will take `inp`. The caller's question, because a caller that gets
        False has to do something else -- vLLM falls back to another all-reduce path."""
        return not self.disabled and self._admits_all_reduce(inp)

    def should_allgather(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._admits_all_gather(inp)

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
        """Enter around a cudagraph capture. NOT optional, and now not skippable in silence: a
        captured launch records an address that is not valid yet, so a backend has to be told.

        The flag is kept here rather than by each backend because it is what makes the omission
        detectable -- see `_check_capture`."""
        self._capturing = True
        try:
            with self._on_capture():
                yield
        finally:
            self._capturing = False

    # ---- The two rules a caller can get wrong, enforced once. ----

    def _check_capture(self, op: str) -> None:
        """Refuse a collective recorded into a graph outside `capture()`.

        Two facts observed independently: the stream says it is capturing, and this object says
        nobody entered the context. They cannot both be right, and the caller is the one that can
        be wrong. Raising HERE names the line; hip's version of getting this wrong was
        `Memory access fault ... on address 0x2000` on all 8 ranks at replay, half an hour later,
        with a null peer pointer as the only clue."""
        if torch.cuda.is_current_stream_capturing() and not self._capturing:
            raise RuntimeError(
                f"{type(self).__name__}.{op} is being captured into a cudagraph, but "
                f"`capture()` was never entered. Wrap the capture: "
                f"`with comm.capture(), torch.cuda.graph(g): ...`. A backend defers work until that "
                f"context exits (hip registers peer pointers there), so a graph captured without it "
                f"replays against addresses that were never registered."
            )

    def _rejected(self, op: str, inp: torch.Tensor) -> str:
        return (f"{type(self).__name__} rejected {op}: shape={tuple(inp.shape)} dtype={inp.dtype} "
                f"disabled={self.disabled}. Ask should_{op.replace('_', '')} first and fall back "
                f"when it says no.")

    # ---- What a BACKEND supplies. ----

    @abstractmethod
    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """SUM across ranks, out of place: input untouched, new tensor returned."""

    @abstractmethod
    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """The per-rank inputs concatenated along `dim`, rank-ordered."""

    def _admits_all_reduce(self, inp: torch.Tensor) -> bool:
        """Constraints beyond `disabled`. Default NONE, which is the honest answer for a backend
        with no hardware limits; `KernelCommunicator` supplies the ones a kernel has."""
        return True

    def _admits_all_gather(self, inp: torch.Tensor) -> bool:
        return True

    def _on_capture(self) -> AbstractContextManager[None]:
        """What this backend must do around a capture. Default nothing, stated by supplying nothing
        rather than by writing an empty override -- iris's empty override set a flag no line read,
        and it was indistinguishable from a real one."""
        return nullcontext()


class KernelCommunicator(Communicator):
    """A `Communicator` whose collectives are a KERNEL, with the admission rules that implies.

    It exists because `IrisCommunicator` and `HipCommunicator` carried these five checks as two
    near-identical functions, under a comment saying the two MUST agree ("two backends that accept
    different tensors are not comparable, and the whole reason this one exists is to be measured
    against that one"). A rule that must hold in two places, enforced by a comment, is one edit away
    from being false -- so it holds in one place now and the requirement is the class hierarchy.

    ALL_REDUCE ONLY. The all_gather rules are NOT shared, and writing this made that visible: iris
    refuses by slab size and ignores dtype and contiguity, hip refuses by dtype and contiguity and
    ignores size. So each backend still answers `_admits_all_gather` for itself, and the fact that
    they disagree is now a difference you can see rather than one hidden in two similar functions.
    (Flagged 2026-08-18: at (4, 8192) bf16 both admit, so it has never bitten.)
    """

    max_size: int
    _SUPPORTED_DTYPES: list

    def _admits_all_reduce(self, inp: torch.Tensor) -> bool:
        """What ANY kernel over a fixed staging buffer needs: a contiguous run of bytes, 16-byte
        aligned for vector loads, inside the buffer, in a dtype the instantiations cover."""
        if not _is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        return True


class IrisCommunicator(KernelCommunicator):
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

    def _admits_all_reduce(self, inp: torch.Tensor) -> bool:
        # The shared kernel rules, plus the two facts only iris has: a live symmetric heap, and
        # input+output having to fit in it.
        if self._shmem is None:
            return False
        if not super()._admits_all_reduce(inp):
            return False
        return inp.numel() * inp.element_size() * 2 <= self._HEAP_SIZE

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

    def _admits_all_gather(self, inp: torch.Tensor) -> bool:
        """Replace every NCCL all_gather; only slab capacity falls back."""
        if self._shmem is None:
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

    # No `_on_capture`: iris needs nothing around a capture, and now says so by supplying nothing.
    # It used to override `capture()` to set `self._IS_CAPTURING`, which NO line in this file reads --
    # copied from `custom_all_reduce.py`, where the flag does have readers. An override that does
    # nothing was indistinguishable from hip's, which does the thing the whole hook exists for.


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

    # No `_admits_*` and no `_on_capture`: torch.distributed has no size, alignment or dtype limit
    # this needs to express, and its collectives need no capture handling. Both are stated by
    # supplying nothing, which is why the base defaults are what they are.

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



class HipCommunicator(KernelCommunicator):
    """Communicator backed by HIP collectives we own (`hip_comms.cu` in this directory).

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

    The kernel is `hip_comms.cu` beside this file, compiled by `hip_comms.py`. It
    depends on torch and the HIP runtime only -- nothing from aiter's `csrc`, which is
    not ours -- so the whole path from Python down to the launch is code we control.

    It self-disables for the same reasons `IrisCommunicator` does (unsupported arch,
    unsupported world size). A failed COMPILE is not one of them and raises: a missing
    iris is genuine unavailability, but our own source failing to build would leave
    vLLM falling back to its own all-reduce and calling the run READY.
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
        self._comms = hip_comms.HipComms(cpu_group, self.device)
        self.disabled = False
        logger.info(
            "HipCommunicator ready: world_size=%d max_size=%dMB",
            self.world_size,
            self.max_size >> 20,
        )

    def _admits_all_reduce(self, inp: torch.Tensor) -> bool:
        # The shared kernel rules come from `KernelCommunicator`, which is what now GUARANTEES the
        # thing this comment used to ask for -- that iris and hip admit the same tensors, since two
        # backends accepting different inputs are not comparable and comparison is why hip exists.
        if not super()._admits_all_reduce(inp):
            return False
        # A two-stage reduce-scatter splits the buffer across ranks, so a count
        # that does not divide is a correctness hazard rather than a slow path.
        return inp.numel() % self.world_size == 0

    def _admits_all_gather(self, inp: torch.Tensor) -> bool:
        # NOT the shared all_reduce rules: a gather has no staging buffer to overflow, so size does
        # not gate it. Deliberately different from iris, which gates on slab size and ignores these
        # two -- see `KernelCommunicator`.
        return _is_weak_contiguous(inp) and inp.dtype in self._SUPPORTED_DTYPES

    def _all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Sum `inp` across the TP ranks. The launch config is chosen in `hip_comms`."""
        out = torch.empty_like(inp)
        self._comms.all_reduce(out, inp)
        return out

    def _all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate every rank's `inp` along `dim`, rank-ordered."""
        return self._comms.all_gather(inp.contiguous(), dim)

    def _on_capture(self) -> AbstractContextManager[None]:
        # The whole reason the hook exists: a captured input's address is not registered
        # when the launch is recorded, so the context reserves a slot during capture and
        # exchanges the IPC handles for everything recorded on the way out. Skipping it left the
        # peer-pointer slot null and faulted all 8 ranks at replay, which is what `_check_capture`
        # now refuses up front.
        return self._comms.capture()


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
    reference/control. The caller (vLLM) stays backend-agnostic and passes nothing; the
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
