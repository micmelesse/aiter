# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Python interface to our HIP collectives.

Two files, both ours: this and `hip_comms.cu`. `torch.utils.cpp_extension.load`
compiles the `.cu` with hipcc and caches the `.so`, so there is no build system, no
entry in aiter's `optCompilerConfig.json`, and nothing in aiter's `csrc`.

`load()` is EAGER by contract -- the caller must call it before entering a cudagraph
capture, because compiling inside capture is not recoverable. `HipCommunicator.__init__`
is where that happens.

Two environment knobs, both set by the arm's Dockerfile rather than defaulted here:
`TORCH_EXTENSIONS_DIR` (where the `.so` is cached; must be an image path, not the
mounted `~/.cache`) and `PYTORCH_ROCM_ARCH` (which torch turns into `--offload-arch`;
required when warming the build with no GPU present to detect).
"""

import threading
from pathlib import Path
from typing import Any

import torch

# The extension's import name. Only used for the build cache and error messages: the
# module object is held here, never imported by name from anywhere else.
NAME = "aiter_hip_comms"

SOURCE = Path(__file__).resolve().parent / "hip_comms.cu"

_module: Any = None
_lock = threading.Lock()


def load() -> Any:
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

        _module = _cpp_load(name=NAME, sources=[str(SOURCE)])
        return _module


def all_reduce(out: torch.Tensor, inp: torch.Tensor) -> None:
    """Sum `inp` across every rank into `out`, in place."""
    load().all_reduce(out, inp)


def all_gather(out: torch.Tensor, inp: torch.Tensor) -> None:
    """Concatenate every rank's `inp` into `out`, rank-ordered, in place."""
    load().all_gather(out, inp)
