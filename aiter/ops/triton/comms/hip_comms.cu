// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained on purpose: torch + the HIP runtime, nothing
// from aiter's csrc, no build system. hip_comms.py compiles this file.
//
// v1 is a STUB that writes zeros. It exists to prove the build and launch path --
// hipcc, the pybind symbol, the op load, the launch, the profile entry -- before any
// of the algorithm exists. Output is deliberately garbage, so no timing from it is an
// all-reduce number.

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

// Grid-stride so the grid is bounded and cannot overflow a launch dim.
// Byte-wise so there is no dtype dispatch here to throw away when the real kernel
// lands. Named to be obvious in a profile.
__global__ void hip_comms_zero_stub(unsigned char* out, int64_t nbytes)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nbytes;
         i += stride)
    {
        out[i] = 0;
    }
}

constexpr int kBlock  = 256;
constexpr int kMaxGrid = 4096;

// Launch ONLY: no hipDeviceSynchronize, no hipMalloc, no host readback. vLLM captures
// a cudagraph around these calls and any of those breaks capture. (The caller's
// torch.empty_like is fine -- it goes through torch's caching allocator, which is
// capture-aware, and IrisCommunicator does the same.)
void launch_zero_stub(torch::Tensor& out)
{
    TORCH_CHECK(out.is_cuda(), "out must be on device");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const int64_t nbytes = out.numel() * out.element_size();
    if (nbytes == 0) return;
    const int grid = static_cast<int>(
        std::min<int64_t>((nbytes + kBlock - 1) / kBlock, kMaxGrid));
    hip_comms_zero_stub<<<grid, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<unsigned char*>(out.data_ptr()), nbytes);
}

void check_same(const torch::Tensor& out, const torch::Tensor& inp)
{
    TORCH_CHECK(inp.is_cuda(), "inp must be on device");
    TORCH_CHECK(inp.is_contiguous(), "inp must be contiguous");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type(),
                "out and inp must have the same dtype");
    TORCH_CHECK(out.device() == inp.device(), "out and inp must be on the same device");
}

}  // namespace

// Both collectives write the whole of `out` and read nothing, so one kernel serves
// both; they stay separate entry points because the caller's intent is what the torch
// trace records. `inp` is validated and otherwise unused until the algorithm exists.
void all_reduce(torch::Tensor& out, torch::Tensor& inp)
{
    check_same(out, inp);
    TORCH_CHECK(out.sizes() == inp.sizes(), "all_reduce: out and inp must have the same shape");
    launch_zero_stub(out);
}

void all_gather(torch::Tensor& out, torch::Tensor& inp)
{
    check_same(out, inp);
    TORCH_CHECK(inp.numel() > 0, "all_gather: inp must be non-empty");
    TORCH_CHECK(out.numel() % inp.numel() == 0,
                "all_gather: out.numel() must be a multiple of inp.numel()");
    launch_zero_stub(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce", &all_reduce, "all-reduce (v1: zero-filling stub)",
          py::arg("out"), py::arg("inp"));
    m.def("all_gather", &all_gather, "all-gather (v1: zero-filling stub)",
          py::arg("out"), py::arg("inp"));
    m.attr("IS_STUB") = true;
}
