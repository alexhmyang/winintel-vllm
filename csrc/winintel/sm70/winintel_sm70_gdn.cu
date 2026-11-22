// SPDX-License-Identifier: Apache-2.0
// Author: 数萌AI服务器 (winintel)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {
__device__ float warp_sum(float x) {
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}

__global__ void convolution(const half* x, const half* weights, half* cache,
                            half* y, const int* starts, const int* ids,
                            const bool* initial, int channels, int64_t xs,
                            int64_t cs, int64_t hs, int64_t ids_stride,
                            bool hf_rounding, const int* accepted) {
  int seq = blockIdx.y, channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  int page = ids[seq * ids_stride], begin = starts[seq], end = starts[seq + 1];
  if (page < 0 || begin == end) return;
  half* history = cache + page * cs + channel;
  float h0 = 0, h1 = 0, h2 = 0;
  if (!initial || initial[seq]) {
    int shift = accepted ? accepted[seq] - 1 : 0;
    h0 = __half2float(history[shift * hs]);
    h1 = __half2float(history[(shift + 1) * hs]);
    h2 = __half2float(history[(shift + 2) * hs]);
  }
  if (accepted) {
    history[0] = __float2half_rn(h1);
    history[hs] = __float2half_rn(h2);
  }
  float w0 = __half2float(weights[channel * 4]);
  float w1 = __half2float(weights[channel * 4 + 1]);
  float w2 = __half2float(weights[channel * 4 + 2]);
  float w3 = __half2float(weights[channel * 4 + 3]);
  for (int t = begin; t < end; ++t) {
    float value = __half2float(x[t * xs + channel]);
    float out = h0 * w0 + h1 * w1 + h2 * w2 + value * w3;
    if (hf_rounding) out = __half2float(__float2half_rn(out));
    y[t * channels + channel] = __float2half_rn(out / (1 + expf(-out)));
    h0 = h1;
    h1 = h2;
    h2 = value;
    if (accepted) history[(t - begin + 2) * hs] = __float2half_rn(value);
  }
  if (!accepted) {
    history[0] = __float2half_rn(h0);
    history[hs] = __float2half_rn(h1);
    history[2 * hs] = __float2half_rn(h2);
  }
}

__global__ void recurrence(const half* qkv, const half* a, const half* b,
                           const half* alog, const half* dt, float* cache,
                           half* out, const int* starts, const int* ids,
                           const bool* initial, int kh, int vh, int64_t cs,
                           int64_t os, int64_t ids_stride, bool hf_rounding,
                           const int* accepted) {
  int seq = blockIdx.x, head = blockIdx.y;
  int lane = threadIdx.x & 31, column = blockIdx.z * 4 + threadIdx.x / 32;
  int slot = accepted ? accepted[seq] - 1 : 0;
  int page = ids[seq * ids_stride + slot];
  int begin = starts[seq], end = starts[seq + 1];
  if (page < 0 || begin == end) return;
  float state[4];
  int64_t offset = page * cs + (head * 128 + column) * 128;
  bool load = !initial || initial[seq];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    state[j] = load ? cache[offset + lane + j * 32] : 0;
  int channels = (2 * kh + vh) * 128, khead = head / (vh / kh);
  for (int t = begin; t < end; ++t) {
    float q[4], k[4], qsum = 0, ksum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int ki = khead * 128 + lane + j * 32;
      q[j] = __half2float(qkv[t * channels + ki]);
      k[j] = __half2float(qkv[t * channels + kh * 128 + ki]);
      qsum += q[j] * q[j];
      ksum += k[j] * k[j];
    }
    float qscale = rsqrtf(warp_sum(qsum) + 1e-6f) * rsqrtf(128.f);
    float kscale = rsqrtf(warp_sum(ksum) + 1e-6f);
    float av = __half2float(a[t * vh + head]) + __half2float(dt[head]);
    float softplus = av > 20 ? av : log1pf(expf(av));
    float decay = expf(-expf(__half2float(alog[head])) * softplus);
    float beta = 1 / (1 + expf(-__half2float(b[t * vh + head])));
    if (hf_rounding) beta = __half2float(__float2half_rn(beta));
    float pred = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q[j] *= qscale;
      k[j] *= kscale;
      state[j] *= decay;
      pred += state[j] * k[j];
    }
    float value =
        __half2float(qkv[t * channels + 2 * kh * 128 + head * 128 + column]);
    float delta = (value - warp_sum(pred)) * beta, result = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      state[j] += delta * k[j];
      result += state[j] * q[j];
    }
    result = warp_sum(result);
    if (lane == 0) out[t * os + head * 128 + column] = __float2half_rn(result);
    if (accepted) {
      int target = ids[seq * ids_stride + t - begin];
      if (target >= 0) {
        int64_t dest = target * cs + (head * 128 + column) * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          cache[dest + lane + j * 32] = state[j];
      }
    }
  }
  if (!accepted) {
#pragma unroll
    for (int j = 0; j < 4; ++j) cache[offset + lane + j * 32] = state[j];
  }
}

__global__ void gated_norm(const half* x, const half* z, const half* w,
                           half* out, int64_t xs, int64_t zs, float eps) {
  int row = blockIdx.x, lane = threadIdx.x;
  float value = __half2float(x[row * xs + lane]);
  float sum = warp_sum(value * value);
  __shared__ float partial[4];
  if ((lane & 31) == 0) partial[lane / 32] = sum;
  __syncthreads();
  float variance = (partial[0] + partial[1] + partial[2] + partial[3]) / 128;
  float gate = __half2float(z[row * zs + lane]);
  out[row * 128 + lane] =
      __float2half_rn(value * rsqrtf(variance + eps) * __half2float(w[lane]) *
                      gate / (1 + expf(-gate)));
}

void check(const torch::Tensor& t, torch::ScalarType dtype,
           const torch::Tensor& x) {
  TORCH_CHECK(
      t.is_cuda() && t.device() == x.device() && t.scalar_type() == dtype,
      "GDN requires tensors of the expected dtype on the same GPU");
}
}  // namespace

void forward(torch::Tensor x, torch::Tensor weights, torch::Tensor conv,
             torch::Tensor ssm, torch::Tensor a, torch::Tensor b,
             torch::Tensor alog, torch::Tensor dt, torch::Tensor starts,
             torch::Tensor ids, c10::optional<torch::Tensor> initial,
             torch::Tensor out, int64_t kh, bool hf_rounding,
             c10::optional<torch::Tensor> accepted) {
  c10::cuda::CUDAGuard guard(x.device());
  for (auto t : {x, weights, conv, a, b, alog, dt, out})
    check(t, torch::kFloat16, x);
  check(ssm, torch::kFloat32, x);
  check(starts, torch::kInt32, x);
  check(ids, torch::kInt32, x);
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "Invalid QKV layout");
  TORCH_CHECK(ssm.dim() == 4 && ssm.size(2) == 128 && ssm.size(3) == 128 &&
                  ssm.stride(3) == 1 && ssm.stride(2) == 128 &&
                  ssm.stride(1) == 16384,
              "Expected [pages, heads, V128, K128] state");
  int vh = ssm.size(1), channels = (2 * kh + vh) * 128;
  TORCH_CHECK(kh > 0 && vh % kh == 0 && x.size(1) == channels,
              "Invalid GDN head dimensions");
  TORCH_CHECK(conv.dim() == 3 && conv.size(0) == ssm.size(0) &&
                  conv.size(1) >= 3 && conv.size(2) == channels &&
                  conv.stride(2) == 1,
              "Expected [pages, history3, channels] convolution");
  TORCH_CHECK(weights.is_contiguous() && weights.numel() == channels * 4,
              "Expected contiguous convolution width four");
  TORCH_CHECK(a.dim() == 2 && a.sizes() == b.sizes() && a.is_contiguous() &&
                  b.is_contiguous() && a.size(0) == x.size(0) &&
                  a.size(1) == vh,
              "Invalid gating layout");
  TORCH_CHECK(alog.is_contiguous() && dt.is_contiguous() &&
                  alog.numel() == vh && dt.numel() == vh,
              "Invalid gating weights");
  TORCH_CHECK(starts.dim() == 1 && starts.is_contiguous() &&
                  (ids.dim() == 1 || ids.dim() == 2) &&
                  (ids.dim() == 1 || ids.stride(-1) == 1) &&
                  starts.numel() == ids.size(0) + 1,
              "Invalid sequence metadata");
  TORCH_CHECK(out.dim() == 3 && out.size(0) >= x.size(0) && out.size(1) == vh &&
                  out.size(2) == 128 && out.stride(2) == 1 &&
                  out.stride(1) == 128,
              "Invalid output layout");
  const bool* flags = nullptr;
  if (initial) {
    check(*initial, torch::kBool, x);
    TORCH_CHECK(initial->is_contiguous() && initial->numel() == ids.size(0),
                "Invalid initial state flags");
    flags = initial->data_ptr<bool>();
  }
  const int* accepted_ptr = nullptr;
  if (accepted) {
    check(*accepted, torch::kInt32, x);
    TORCH_CHECK(accepted->is_contiguous() &&
                    accepted->numel() == ids.size(0) && ids.dim() == 2 &&
                    conv.size(1) >= ids.size(1) + 2,
                "Invalid speculative state metadata");
    accepted_ptr = accepted->data_ptr<int>();
  } else {
    TORCH_CHECK(ids.dim() == 1, "Non-speculative states require 1D indices");
  }
  int sequences = ids.size(0);
  if (!sequences) return;
  auto y = torch::empty({x.size(0), channels}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  convolution<<<dim3((channels + 127) / 128, sequences), 128, 0, stream>>>(
      reinterpret_cast<half*>(x.data_ptr()),
      reinterpret_cast<half*>(weights.data_ptr()),
      reinterpret_cast<half*>(conv.data_ptr()),
      reinterpret_cast<half*>(y.data_ptr()), starts.data_ptr<int>(),
      ids.data_ptr<int>(), flags, channels, x.stride(0), conv.stride(0),
      conv.stride(1), ids.stride(0), hf_rounding, accepted_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  recurrence<<<dim3(sequences, vh, 32), 128, 0, stream>>>(
      reinterpret_cast<half*>(y.data_ptr()),
      reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b.data_ptr()),
      reinterpret_cast<half*>(alog.data_ptr()),
      reinterpret_cast<half*>(dt.data_ptr()), ssm.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr()), starts.data_ptr<int>(),
      ids.data_ptr<int>(), flags, kh, vh, ssm.stride(0), out.stride(0),
      ids.stride(0), hf_rounding, accepted_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor sm70_norm(torch::Tensor x, torch::Tensor z, torch::Tensor w,
                        double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  for (auto t : {x, z, w}) check(t, torch::kFloat16, x);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128 && z.sizes() == x.sizes() &&
                  x.stride(1) == 1 && z.stride(1) == 1 && w.is_contiguous() &&
                  w.numel() == 128,
              "Invalid gated RMSNorm layout");
  auto out = torch::empty_like(x);
  if (x.size(0))
    gated_norm<<<x.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(x.data_ptr()),
        reinterpret_cast<half*>(z.data_ptr()),
        reinterpret_cast<half*>(w.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()), x.stride(0), z.stride(0), eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, pybind11::arg("x"), pybind11::arg("weights"),
        pybind11::arg("conv"), pybind11::arg("ssm"), pybind11::arg("a"),
        pybind11::arg("b"), pybind11::arg("alog"), pybind11::arg("dt"),
        pybind11::arg("starts"), pybind11::arg("ids"), pybind11::arg("initial"),
        pybind11::arg("out"), pybind11::arg("kh"),
        pybind11::arg("hf_rounding") = false,
        pybind11::arg("accepted") = c10::nullopt);
  m.def("norm", &sm70_norm);
}
