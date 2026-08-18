"""Every backend admits exactly what `CustomAllreduce` admits.

This is the property that makes the comms experiment mean anything. The backends REPLACE
CustomAllreduce, so if one admits a tensor the baseline declines (or the reverse) the two arms route
different work to different code and the measurement compares the routing, not the kernels. It went
wrong once already: bounding all_gather by `max_size` instead of `max_size / (world_size * 2)` made
us 16x looser than baseline at TP=8, and bounding all_reduce by `max_size` instead of the decode
ceiling made us 8x tighter -- so any prefill batch over 512 tokens fell back for us and did not for
baseline.

No GPU: both predicates read only a tensor's metadata plus a few instance attributes, so the
instances are built with `object.__new__` and CPU tensors are enough.

KNOWN_DIVERGENCES may only SHRINK. A new disagreement fails, and an entry that stops diverging fails
too, so nobody can leave a stale exemption behind.
"""

import itertools
import sys

import pytest
import torch

sys.path.insert(0, __file__.rsplit("/op_tests/", 1)[0])

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.ops.triton.comms.communicator import TorchCommunicator

WORLD_SIZES = (2, 4, 8)
MAX_SIZE = 8 << 20
DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# The ONE envelope difference we accept, with the reason it is not a choice: `hip_comms.cu` has
# instantiations for fp16 and bf16 only, so an fp32 all-reduce falls back for us and does not for
# baseline. Closing it means adding a kernel instantiation, not editing a predicate.
KNOWN_DIVERGENCES = frozenset({"dtype=torch.float32"})


def _baseline(world: int) -> CustomAllreduce:
    b = object.__new__(CustomAllreduce)
    b.disabled = False
    b.world_size = world
    b.fully_connected = True      # an 8xMI350 box; the gate this stands in for is topology
    b.max_size = MAX_SIZE
    return b


def _ours(world: int) -> TorchCommunicator:
    c = object.__new__(TorchCommunicator)
    c.disabled = False
    c.world_size = world
    c.max_size = MAX_SIZE
    return c


def _label(dtype: torch.dtype) -> str:
    return f"dtype={dtype}"


def _tensors(dtype: torch.dtype):
    """Sizes chosen to sit ON the boundaries, where an off-by-one hides: each bound, one element
    either side of it, and the alignment rule."""
    element = torch.empty(0, dtype=dtype).element_size()
    edges = {1, 8, 16, 512, 1 << 20, MAX_SIZE // 32, MAX_SIZE // 16, MAX_SIZE // 8,
             MAX_SIZE, 8192 * 8192, 8192 * 8192 + 16, 1 << 27}
    for nbytes in sorted({n + d for n in edges for d in (-element, 0, element) if n + d > 0}):
        if nbytes % element == 0:
            yield torch.empty(nbytes // element, dtype=dtype)


@pytest.mark.parametrize("world,dtype", list(itertools.product(WORLD_SIZES, DTYPES)),
                         ids=lambda v: str(v))
def test_all_reduce_admission_matches_the_baseline(world: int, dtype: torch.dtype) -> None:
    base, ours = _baseline(world), _ours(world)
    for t in _tensors(dtype):
        want, got = base.should_custom_ar(t), ours.should_allreduce(t)
        if want == got:
            continue
        assert _label(dtype) in KNOWN_DIVERGENCES, (
            f"all_reduce admission diverged from CustomAllreduce: world={world} dtype={dtype} "
            f"nbytes={t.numel() * t.element_size()} baseline={want} ours={got}. The arms now route "
            f"different work, so the comparison measures routing rather than kernels."
        )


@pytest.mark.parametrize("world,dtype", list(itertools.product(WORLD_SIZES, DTYPES)),
                         ids=lambda v: str(v))
def test_all_gather_admission_matches_the_baseline(world: int, dtype: torch.dtype) -> None:
    base, ours = _baseline(world), _ours(world)
    for t in _tensors(dtype):
        want, got = base.should_custom_ag(t), ours.should_allgather(t)
        if want == got:
            continue
        assert _label(dtype) in KNOWN_DIVERGENCES, (
            f"all_gather admission diverged from CustomAllreduce: world={world} dtype={dtype} "
            f"nbytes={t.numel() * t.element_size()} baseline={want} ours={got}."
        )


def test_every_declared_divergence_still_diverges() -> None:
    """A shrink-only ratchet: an exemption that no longer diverges is a stale one, and a stale
    exemption is how a real divergence gets waved through later."""
    stale = set()
    for dtype in DTYPES:
        if _label(dtype) not in KNOWN_DIVERGENCES:
            continue
        base, ours = _baseline(8), _ours(8)
        if all(base.should_custom_ar(t) == ours.should_allreduce(t) for t in _tensors(dtype)):
            stale.add(_label(dtype))
    assert not stale, f"no longer diverging, so delete from KNOWN_DIVERGENCES: {sorted(stale)}"


def test_the_admission_rules_are_the_SAME_for_every_backend() -> None:
    """The other half: uniform ACROSS backends. `Communicator` owns admission and
    `__init_subclass__` refuses an override, so this asserts the mechanism rather than each pair."""
    from aiter.ops.triton.comms.communicator import (Communicator, HipCommunicator,
                                                     IrisCommunicator)

    for cls in (TorchCommunicator, IrisCommunicator, HipCommunicator):
        for name in ("should_allreduce", "should_allgather", "_shaped_for_a_kernel"):
            assert name not in cls.__dict__, f"{cls.__name__} redefines {name}"
            assert getattr(cls, name) is getattr(Communicator, name)
