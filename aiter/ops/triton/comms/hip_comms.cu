// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained on purpose: torch + the HIP runtime, nothing
// from aiter's csrc, no build system. hip_comms.py compiles this file.

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>

namespace {

// CURRENT IMPLEMENTATION: this writes zeros. The two-stage reduce-scatter/all-gather
// is not written yet, and this is the only place that knows it -- nothing upstream
// branches on it.
//
// Grid-stride so the grid is bounded and cannot overflow a launch dim. Byte-wise so
// there is no dtype dispatch to unpick when the real algorithm lands.
__global__ void hip_comms_fill_zero(unsigned char* out, int64_t nbytes)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nbytes;
         i += stride)
    {
        out[i] = 0;
    }
}

constexpr int kBlock   = 256;
constexpr int kMaxGrid = 4096;

// Launch ONLY: no hipDeviceSynchronize, no hipMalloc, no host readback. vLLM captures
// a cudagraph around these calls and any of those breaks capture. (The caller's
// torch.empty_like is fine -- it goes through torch's caching allocator, which is
// capture-aware, and IrisCommunicator does the same.)
void launch(torch::Tensor& out)
{
    TORCH_CHECK(out.is_cuda(), "out must be on device");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const int64_t nbytes = out.numel() * out.element_size();
    if (nbytes == 0) return;
    const int grid = static_cast<int>(
        std::min<int64_t>((nbytes + kBlock - 1) / kBlock, kMaxGrid));
    hip_comms_fill_zero<<<grid, kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<unsigned char*>(out.data_ptr()), nbytes);
}

void check_pair(const torch::Tensor& out, const torch::Tensor& inp)
{
    TORCH_CHECK(inp.is_cuda(), "inp must be on device");
    TORCH_CHECK(inp.is_contiguous(), "inp must be contiguous");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type(),
                "out and inp must have the same dtype");
    TORCH_CHECK(out.device() == inp.device(), "out and inp must be on the same device");
}

}  // namespace

// Sum `inp` across every rank into `out`.
void all_reduce(torch::Tensor& out, torch::Tensor& inp)
{
    check_pair(out, inp);
    TORCH_CHECK(out.sizes() == inp.sizes(),
                "all_reduce: out and inp must have the same shape");
    launch(out);
}

// Concatenate every rank's `inp` into `out`, rank-ordered.
void all_gather(torch::Tensor& out, torch::Tensor& inp)
{
    check_pair(out, inp);
    TORCH_CHECK(inp.numel() > 0, "all_gather: inp must be non-empty");
    TORCH_CHECK(out.numel() % inp.numel() == 0,
                "all_gather: out.numel() must be a multiple of inp.numel()");
    launch(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("all_reduce", &all_reduce, "all-reduce across the TP ranks",
          py::arg("out"), py::arg("inp"));
    m.def("all_gather", &all_gather, "all-gather across the TP ranks",
          py::arg("out"), py::arg("inp"));
}
