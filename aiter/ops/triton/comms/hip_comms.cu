// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Our HIP collectives. Self-contained: torch + the HIP runtime, nothing from aiter's
// csrc, no build system. hip_comms.py compiles this file.
//
// THIS FILE HOLDS NO POLICY. It has no size heuristics, no getenv, and no tuned
// constants: which algorithm, how many blocks and how many threads all arrive as
// arguments, chosen in Python. A decision made here would be invisible and untestable,
// and the escape hatch it eventually needs is how you end up with an env var deciding
// how a kernel runs. What lives here is mechanism -- the peer handshake, the barrier,
// the reduction -- plus a dispatch table over the instantiations that exist.
//
// Compile-time vs runtime is the one distinction that shapes everything. `ngpus` and the
// dtype must be constexpr to unroll and vectorize, so they are template parameters and
// the instantiation list below is the finite menu Python may pick from; asking for one
// that was not built RAISES rather than falling back, because a silent substitution
// produces a number about the wrong thing.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_comms {

constexpr int kMaxRanks  = 8;
constexpr int kMaxBlocks = 36;

#define HIP_CHECK(expr)                                                             \
  do {                                                                              \
    hipError_t _e = (expr);                                                         \
    if (_e != hipSuccess) {                                                         \
      throw std::runtime_error(std::string("hip_comms: ") + #expr + " -> " +         \
                               hipGetErrorString(_e));                              \
    }                                                                               \
  } while (0)

// ---------------------------------------------------------------------------------
// Shared layout. One IPC allocation per rank holds the signal block AND the scratch the
// two-stage algorithm needs, so adding that algorithm changes no layout: scratch is
// simply the bytes after the struct.
// ---------------------------------------------------------------------------------

// TWO counter arrays, not one. A peer block can reach the second barrier while this one
// is still at the first, and with a single array the peer would write counter+1 while we
// busy-wait on counter. `seq` is the per-block monotonic sequence number.
struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t end[kMaxBlocks][kMaxRanks];
  alignas(128) uint32_t seq[kMaxBlocks];
};

struct __align__(16) PeerPtrs { void* p[kMaxRanks]; };
struct __align__(16) PeerSignals { Signal* s[kMaxRanks]; };

__device__ __forceinline__ void* scratch_of(Signal* sig) { return sig + 1; }

#define DINLINE __device__ __forceinline__

// ---------------------------------------------------------------------------------
// Barrier. System scope to REACH a peer's memory, device scope to poll our own.
// ---------------------------------------------------------------------------------

template <int ngpus>
DINLINE void barrier_start(PeerSignals sigs, Signal* self, int rank) {
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->start[blockIdx.x][rank], f,
                            __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < f);
  }
  __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// `final_sync` drops the release/acquire pair: nothing after the last barrier reads what
// this kernel wrote, so ordering costs without buying anything.
template <int ngpus, bool final_sync>
DINLINE void barrier_end(PeerSignals sigs, Signal* self, int rank) {
  __syncthreads();
  uint32_t f = self->seq[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sigs.s[threadIdx.x]->end[blockIdx.x][rank], f,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self->end[blockIdx.x][threadIdx.x],
                                  final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_DEVICE) < f);
  }
  if constexpr (!final_sync) __syncthreads();
  if (threadIdx.x == 0) self->seq[blockIdx.x] = f;
}

// ---------------------------------------------------------------------------------
// Vectorized reduce. 16 bytes per thread, accumulated in fp32 so a bf16 sum of 8 values
// rounds once at the end rather than eight times along the way.
// ---------------------------------------------------------------------------------

template <typename T, int N>
struct __align__(sizeof(T) * N) vec {
  T d[N];
};

template <typename T>
struct traits {
  static constexpr int N = 16 / sizeof(T);
  using V = vec<T, N>;
};

template <typename T, int ngpus>
DINLINE typename traits<T>::V reduce_at(const typename traits<T>::V* const ptrs[],
                                        int idx) {
  constexpr int N = traits<T>::N;
  float acc[N];
  auto v0 = ptrs[0][idx];
#pragma unroll
  for (int j = 0; j < N; ++j) acc[j] = static_cast<float>(v0.d[j]);
#pragma unroll
  for (int i = 1; i < ngpus; ++i) {
    auto v = ptrs[i][idx];
#pragma unroll
    for (int j = 0; j < N; ++j) acc[j] += static_cast<float>(v.d[j]);
  }
  typename traits<T>::V out;
#pragma unroll
  for (int j = 0; j < N; ++j) out.d[j] = static_cast<T>(acc[j]);
  return out;
}

// ---------------------------------------------------------------------------------
// Kernels. One per algorithm; `algo` selects among them at dispatch.
// ---------------------------------------------------------------------------------

// ONE-SHOT: every rank reads every peer's input over the whole buffer and writes its own
// output. Moves ngpus x the bytes of a two-stage, so it is the wrong algorithm at 1 MiB
// and the right one to land first: it exercises the peer handshake end to end and is
// correct, which makes two-stage a pure performance change against a working baseline.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    one_shot_all_reduce(const PeerPtrs* peers, PeerSignals sigs, Signal* self,
                        T* __restrict__ out, int rank, int size) {
  using V = typename traits<T>::V;
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i)
    ptrs[i] = reinterpret_cast<const V*>(peers->p[(rank + i) % ngpus]);

  barrier_start<ngpus>(sigs, self, rank);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = tid; idx < size; idx += stride)
    reinterpret_cast<V*>(out)[idx] = reduce_at<T, ngpus>(ptrs, idx);
  // Required, not defensive: without it a rank can return and let its INPUT be reused
  // while a peer is still reading that input.
  barrier_end<ngpus, true>(sigs, self, rank);
}

// ONE-SHOT ALL-GATHER. Same handshake and same read pattern as the all-reduce; it
// concatenates instead of summing. `out` is filled RANK-MAJOR -- shape (ngpus, *inp) --
// and the Python layer moves the axis where the caller wanted it, exactly as the torch
// path did, so the two produce identical results.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    one_shot_all_gather(const PeerPtrs* peers, PeerSignals sigs, Signal* self,
                        T* __restrict__ out, int rank, int size) {
  using V = typename traits<T>::V;
  const V* ptrs[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; ++i) ptrs[i] = reinterpret_cast<const V*>(peers->p[i]);

  barrier_start<ngpus>(sigs, self, rank);
  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  V* dst           = reinterpret_cast<V*>(out);
  for (int idx = tid; idx < size; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; ++i) dst[i * size + idx] = ptrs[i][idx];
  }
  barrier_end<ngpus, true>(sigs, self, rank);
}

// ---------------------------------------------------------------------------------
// The context: peer memory, registration, dispatch. One per process group.
// ---------------------------------------------------------------------------------

using Handle = hipIpcMemHandle_t;

static Handle handle_from(const std::string& bytes) {
  if (bytes.size() != sizeof(Handle))
    throw std::runtime_error("hip_comms: ipc handle is " + std::to_string(bytes.size()) +
                             " bytes, expected " + std::to_string(sizeof(Handle)));
  Handle h;
  std::memcpy(&h, bytes.data(), sizeof(Handle));
  return h;
}

// hipIpcGetMemHandle must be given the BASE of an allocation, but a torch tensor sits at
// an offset inside one -- so the handle names the allocation and the offset locates the
// tensor within it. Returning both together is what keeps the two from drifting apart.
static hipPointer_attribute range_start_attr = HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;

// py::bytes, NOT std::string: pybind converts std::string to a Python `str` by decoding
// UTF-8, and an IPC handle is arbitrary binary that will not decode. Returning it as a
// string throws UnicodeDecodeError on the first call. The INPUT direction is safe as
// std::string because pybind accepts `bytes` for it.
static std::pair<py::bytes, int64_t> ipc_handle_and_offset(uintptr_t ptr) {
  void* base = nullptr;
  HIP_CHECK(hipPointerGetAttribute(&base, range_start_attr,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  Handle h;
  HIP_CHECK(hipIpcGetMemHandle(&h, base));
  return {py::bytes(reinterpret_cast<const char*>(&h), sizeof(Handle)),
          reinterpret_cast<char*>(ptr) - static_cast<char*>(base)};
}

class Comms {
 public:
  // `signal_handles`/`signal_offsets` are the whole world's handles for their own signal
  // allocation, gathered in PYTHON -- the collective that exchanges them belongs to the
  // process group, which C++ has no business knowing about.
  Comms(int rank, int world_size, uintptr_t self_signal,
        const std::vector<std::string>& signal_handles,
        const std::vector<int64_t>& signal_offsets, uintptr_t peer_slab,
        int64_t peer_slab_bytes)
      : rank_(rank),
        world_size_(world_size),
        self_signal_(reinterpret_cast<Signal*>(self_signal)),
        slab_(reinterpret_cast<PeerPtrs*>(peer_slab)),
        slab_end_(reinterpret_cast<PeerPtrs*>(peer_slab) +
                  peer_slab_bytes / sizeof(PeerPtrs)),
        cursor_(reinterpret_cast<PeerPtrs*>(peer_slab)) {
    if (world_size_ < 2 || world_size_ > kMaxRanks)
      throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                               " outside [2, " + std::to_string(kMaxRanks) + "]");
    if (signal_handles.size() != static_cast<size_t>(world_size_) ||
        signal_offsets.size() != static_cast<size_t>(world_size_))
      throw std::runtime_error("hip_comms: expected one signal handle+offset per rank");
    auto opened = open_peers(signal_handles, signal_offsets, self_signal);
    for (int i = 0; i < world_size_; ++i)
      peer_signals_.s[i] = reinterpret_cast<Signal*>(opened[i]);
  }

  ~Comms() {
    for (const auto& kv : opened_) hipIpcCloseMemHandle(kv.second);
  }

  // A buffer whose address is known ahead of time. The eager path.
  void register_buffer(const std::vector<std::string>& handles,
                       const std::vector<int64_t>& offsets, uintptr_t self_ptr) {
    auto ptrs = open_peers(handles, offsets, self_ptr);
    registered_[reinterpret_cast<void*>(self_ptr)] = commit(ptrs);
  }

  // The CAPTURE path, in two halves. During capture the input address is not registered
  // yet, so `all_reduce` reserves a slab slot and remembers the pointer; afterwards
  // Python gathers handles for everything remembered and this fills the slots in. Sound
  // because a captured address is fixed for the graph's life -- the kernel reads a
  // PeerPtrs populated AFTER the capture that recorded the launch.
  std::vector<uintptr_t> pending_graph_buffers() const {
    std::vector<uintptr_t> out;
    out.reserve(pending_.size());
    for (void* p : pending_) out.push_back(reinterpret_cast<uintptr_t>(p));
    return out;
  }

  void register_graph_buffers(const std::vector<std::vector<std::string>>& handles,
                              const std::vector<std::vector<int64_t>>& offsets) {
    if (handles.size() != pending_.size() || offsets.size() != pending_.size())
      throw std::runtime_error("hip_comms: got handles for " +
                               std::to_string(handles.size()) + " buffers, " +
                               std::to_string(pending_.size()) + " are pending");
    // The slots are filled in; nothing goes into `registered_`. That map means "an address
    // this object keeps alive", and a captured buffer is the opposite -- it dies with its
    // graph. Nothing needs it there either: the slot pointer is baked into the launch the
    // capture recorded, so a replay never looks the address up. `registered_` therefore holds
    // exactly what `register_buffer` was called for, which is what Python's `_registered`
    // tracks.
    for (size_t i = 0; i < pending_.size(); ++i) {
      auto ptrs = open_peers(handles[i], offsets[i],
                             reinterpret_cast<uintptr_t>(pending_[i]));
      write_slot(pending_slots_[i], ptrs);
    }
    pending_.clear();
    pending_slots_.clear();
  }

  int64_t pending_count() const { return static_cast<int64_t>(pending_.size()); }

  // `algo`, `blocks` and `threads` are the CALLER's decision. Nothing here inspects the
  // size to pick them.
  void all_reduce(torch::Tensor& out, torch::Tensor& inp, int64_t algo, int64_t blocks,
                  int64_t threads) {
    TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
    TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
    TORCH_CHECK(out.sizes() == inp.sizes(), "out and inp must have the same shape");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
    TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "blocks must be in [1, ", kMaxBlocks, "]");
    TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

    const int lanes = 16 / static_cast<int>(inp.element_size());
    TORCH_CHECK(inp.numel() % lanes == 0, "numel ", inp.numel(),
                " must be a multiple of ", lanes, " for 16-byte vectorized access");
    const int n = static_cast<int>(inp.numel() / lanes);

    PeerPtrs* slot = slot_for(inp.data_ptr());
    dispatch<false>(out, algo, blocks, threads, slot, n);
  }

  // `out` must be rank-major with ngpus x inp.numel() elements; the caller reshapes.
  void all_gather(torch::Tensor& out, torch::Tensor& inp, int64_t algo, int64_t blocks,
                  int64_t threads) {
    TORCH_CHECK(out.is_cuda() && inp.is_cuda(), "out and inp must be on device");
    TORCH_CHECK(out.is_contiguous() && inp.is_contiguous(), "out and inp must be contiguous");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type(), "out and inp must share a dtype");
    TORCH_CHECK(out.numel() == inp.numel() * world_size_, "all_gather: out must hold ",
                world_size_, " x inp (", inp.numel() * world_size_, "), got ", out.numel());
    TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks, "blocks must be in [1, ", kMaxBlocks, "]");
    TORCH_CHECK(threads > 0 && threads <= 512, "threads must be in [1, 512]");

    const int lanes = 16 / static_cast<int>(inp.element_size());
    TORCH_CHECK(inp.numel() % lanes == 0, "numel ", inp.numel(),
                " must be a multiple of ", lanes, " for 16-byte vectorized access");
    PeerPtrs* slot = slot_for(inp.data_ptr());
    dispatch<true>(out, algo, blocks, threads, slot,
                   static_cast<int>(inp.numel() / lanes));
  }

 private:
  // Open every rank's handle into a local pointer. Our OWN handle is never opened --
  // hipIpcOpenMemHandle refuses a self-handle -- so the local pointer is used directly.
  std::vector<void*> open_peers(const std::vector<std::string>& handles,
                                const std::vector<int64_t>& offsets, uintptr_t self) {
    std::vector<void*> out(world_size_, nullptr);
    for (int i = 0; i < world_size_; ++i) {
      if (i == rank_) {
        out[i] = reinterpret_cast<void*>(self);
        continue;
      }
      // Once per distinct handle. A capture-heavy run exchanges the same peer BASES over
      // and over -- one per graph, per rank -- and hipIpcOpenMemHandle on a handle this
      // process already mapped is not a second mapping to close later. vLLM's
      // CustomAllreduce keeps the same cache, keyed the same way.
      auto it = opened_.find(handles[i]);
      if (it == opened_.end()) {
        void* base = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(&base, handle_from(handles[i]),
                                      hipIpcMemLazyEnablePeerAccess));
        it = opened_.emplace(handles[i], base).first;
      }
      out[i] = static_cast<char*>(it->second) + offsets[i];
    }
    return out;
  }

  PeerPtrs* commit(const std::vector<void*>& ptrs) {
    PeerPtrs* slot = next_slot();
    write_slot(slot, ptrs);
    return slot;
  }

  PeerPtrs* next_slot() {
    if (cursor_ >= slab_end_)
      throw std::runtime_error(
          "hip_comms: peer-pointer slab is full; allocate a larger one");
    return cursor_++;
  }

  void write_slot(PeerPtrs* slot, const std::vector<void*>& ptrs) {
    PeerPtrs host{};
    for (int i = 0; i < world_size_; ++i) host.p[i] = ptrs[i];
    HIP_CHECK(hipMemcpy(slot, &host, sizeof(PeerPtrs), hipMemcpyHostToDevice));
  }

  PeerPtrs* slot_for(void* input) {
    hipStreamCaptureStatus status;
    HIP_CHECK(hipStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    if (status == hipStreamCaptureStatusActive) {
      // A fresh slot ALWAYS, even for an address `registered_` already knows. An address is
      // only as durable as the allocation under it: a graph's buffers are freed when the
      // graph dies and the allocator hands the same address back for the next one. Skipping
      // the record here would make the recorded COUNT depend on that luck -- and what
      // follows a capture is a COLLECTIVE exchange, so ranks that skip differently do not
      // merely disagree, they exchange the wrong number of handles.
      PeerPtrs* slot = next_slot();
      pending_.push_back(input);
      pending_slots_.push_back(slot);
      return slot;
    }
    auto it = registered_.find(input);
    if (it == registered_.end()) {
      std::ostringstream os;
      os << "hip_comms: buffer " << input
         << " is not registered. Register it, or call inside a cudagraph capture (where "
            "registration is deferred until after capture).";
      throw std::runtime_error(os.str());
    }
    return it->second;
  }

  // THE instantiation menu. Every combination that exists is named here exactly once, so
  // an unsupported request is a listed refusal rather than a wrong kernel.
  template <bool gather>
  void dispatch(torch::Tensor& out, int64_t algo, int64_t blocks, int64_t threads,
                PeerPtrs* slot, int n) {
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_ONE_SHOT(T, NG)                                                          \
  do {                                                                                  \
    if constexpr (gather)                                                               \
      one_shot_all_gather<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(           \
          slot, peer_signals_, self_signal_, out.data_ptr<T>(), rank_, n);              \
    else                                                                                \
      one_shot_all_reduce<T, NG><<<dim3(blocks), dim3(threads), 0, stream>>>(           \
          slot, peer_signals_, self_signal_, out.data_ptr<T>(), rank_, n);              \
  } while (0)

#define BY_NGPUS(T)                                                                     \
  switch (world_size_) {                                                                \
    case 2: LAUNCH_ONE_SHOT(T, 2); return;                                              \
    case 4: LAUNCH_ONE_SHOT(T, 4); return;                                              \
    case 8: LAUNCH_ONE_SHOT(T, 8); return;                                              \
    default: break;                                                                     \
  }

    if (algo != kAlgoOneShot)
      throw std::runtime_error("hip_comms: algo " + std::to_string(algo) +
                               " is not built. Built: 0 (one_shot).");
    switch (out.scalar_type()) {
      case at::ScalarType::Half: BY_NGPUS(at::Half) break;
      case at::ScalarType::BFloat16: BY_NGPUS(at::BFloat16) break;
      default:
        throw std::runtime_error("hip_comms: dtype not built. Built: float16, bfloat16.");
    }
    throw std::runtime_error("hip_comms: world_size " + std::to_string(world_size_) +
                             " not built. Built: 2, 4, 8.");
#undef BY_NGPUS
#undef LAUNCH_ONE_SHOT
  }

  static constexpr int64_t kAlgoOneShot = 0;

  int rank_;
  int world_size_;
  Signal* self_signal_;
  PeerSignals peer_signals_{};
  PeerPtrs* slab_;
  PeerPtrs* slab_end_;
  PeerPtrs* cursor_;
  std::unordered_map<void*, PeerPtrs*> registered_;
  std::vector<void*> pending_;
  std::vector<PeerPtrs*> pending_slots_;
  std::unordered_map<std::string, void*> opened_;
};

}  // namespace hip_comms

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using hip_comms::Comms;
  m.attr("SIGNAL_BYTES")   = py::int_(sizeof(hip_comms::Signal));
  m.attr("PEER_PTRS_BYTES") = py::int_(sizeof(hip_comms::PeerPtrs));
  m.attr("MAX_BLOCKS")     = py::int_(hip_comms::kMaxBlocks);
  m.attr("MAX_RANKS")      = py::int_(hip_comms::kMaxRanks);
  m.attr("IPC_HANDLE_BYTES") = py::int_(sizeof(hipIpcMemHandle_t));

  m.def("ipc_handle_and_offset", &hip_comms::ipc_handle_and_offset, py::arg("ptr"),
        "The IPC handle of the allocation containing `ptr`, and `ptr`'s offset in it.");

  py::class_<Comms>(m, "Comms")
      .def(py::init<int, int, uintptr_t, const std::vector<std::string>&,
                    const std::vector<int64_t>&, uintptr_t, int64_t>(),
           py::arg("rank"), py::arg("world_size"), py::arg("self_signal"),
           py::arg("signal_handles"), py::arg("signal_offsets"), py::arg("peer_slab"),
           py::arg("peer_slab_bytes"))
      .def("register_buffer", &Comms::register_buffer, py::arg("handles"),
           py::arg("offsets"), py::arg("self_ptr"))
      .def("pending_graph_buffers", &Comms::pending_graph_buffers)
      .def("register_graph_buffers", &Comms::register_graph_buffers, py::arg("handles"),
           py::arg("offsets"))
      .def("pending_count", &Comms::pending_count)
      .def("all_reduce", &Comms::all_reduce, py::arg("out"), py::arg("inp"),
           py::arg("algo"), py::arg("blocks"), py::arg("threads"))
      .def("all_gather", &Comms::all_gather, py::arg("out"), py::arg("inp"),
           py::arg("algo"), py::arg("blocks"), py::arg("threads"));
}
