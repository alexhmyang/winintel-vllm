// SPDX-License-Identifier: Apache-2.0
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

void check_tensor(const torch::Tensor& tensor, torch::ScalarType dtype,
                  const torch::Tensor& input) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
              "W4 tensors must be contiguous CUDA tensors");
  TORCH_CHECK(tensor.scalar_type() == dtype, "W4 tensor dtype mismatch");
  TORCH_CHECK(tensor.device() == input.device(), "W4 tensor device mismatch");
}

__device__ float sum_warp(float value) {
  for (int d = 16; d; d >>= 1) value += __shfl_down_sync(0xffffffff, value, d);
  return value;
}

__global__ void pack_weights(const half* w, uint16_t* packed, half* scales,
                             int groups, int64_t count) {
  int lane = threadIdx.x & 31;
  int64_t group = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (group >= count) return;
  float values[4], maximum = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = __half2float(w[group * 128 + lane * 4 + i]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  for (int d = 16; d; d >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
  half scale_half = __float2half_rn(maximum / 7.0f);
  float scale = __half2float(scale_half);
  uint16_t bits = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int q = scale > 0 ? __float2int_rn(values[i] / scale) : 0;
    q = max(-7, min(7, q));
    bits |= uint16_t(q + 8) << (i * 4);
  }
  packed[group * 32 + lane] = bits;
  if (!lane) scales[group] = scale_half;
}

template <bool GATE>
__global__ void routed_linear(const half* x, const uint16_t* packed,
                              const half* scales, const int* ids, half* out,
                              int tokens, int topk, int rows, int columns,
                              int experts, bool rounding) {
  int lane = threadIdx.x & 31;
  int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  int output_rows = GATE ? rows / 2 : rows;
  if (index >= int64_t(tokens) * topk * output_rows) return;
  int route = index / output_rows, row = index % output_rows;
  int expert = ids[route];
  float value = 0, up = 0;
  if (expert >= 0 && expert < experts) {
    const half* input = x + int64_t(GATE ? route / topk : route) * columns;
    int64_t base_row = int64_t(expert) * rows + row;
    int groups = columns / 128;
    for (int group = 0; group < groups; ++group) {
      int64_t offset = (base_row * groups + group) * 32 + lane;
      uint16_t bits = packed[offset];
      uint16_t up_bits =
          GATE ? packed[offset + int64_t(output_rows) * groups * 32] : 0;
      half scale = scales[base_row * groups + group];
      half up_scale = GATE ? scales[(base_row + output_rows) * groups + group]
                           : __float2half(0);
      half2 scale_pair = __halves2half2(scale, scale);
      half2 up_scale_pair = __halves2half2(up_scale, up_scale);
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float2 activation = __half22float2(*reinterpret_cast<const half2*>(
            input + group * 128 + lane * 4 + i));
        unsigned int code_bits = (bits >> (i * 4));
        unsigned int magic =
            0x64006400u | (code_bits & 15u) | ((code_bits & 240u) << 12);
        half2 codes = __hsub2(*reinterpret_cast<half2*>(&magic),
                              __float2half2_rn(1032.0f));
        float2 weight = __half22float2(__hmul2(codes, scale_pair));
        value = fmaf(activation.x, weight.x, value);
        value = fmaf(activation.y, weight.y, value);
        if (GATE) {
          unsigned int up_code_bits = (up_bits >> (i * 4));
          unsigned int up_magic = 0x64006400u | (up_code_bits & 15u) |
                                  ((up_code_bits & 240u) << 12);
          half2 up_codes = __hsub2(*reinterpret_cast<half2*>(&up_magic),
                                   __float2half2_rn(1032.0f));
          float2 up_weight = __half22float2(__hmul2(up_codes, up_scale_pair));
          up = fmaf(activation.x, up_weight.x, up);
          up = fmaf(activation.y, up_weight.y, up);
        }
      }
    }
  }
  value = sum_warp(value);
  if (GATE) up = sum_warp(up);
  if (!lane) {
    if (GATE) {
      value = __half2float(__float2half_rn(value));
      up = __half2float(__float2half_rn(up));
      value = value / (1.0f + expf(-value));
      if (rounding) value = __half2float(__float2half_rn(value));
      value *= up;
    }
    out[index] = __float2half_rn(value);
  }
}

void pack(torch::Tensor w, torch::Tensor packed, torch::Tensor scales) {
  check_tensor(w, torch::kHalf, w);
  check_tensor(packed, torch::kUInt16, w);
  check_tensor(scales, torch::kHalf, w);
  TORCH_CHECK(
      w.dim() == 3 && w.size(2) > 0 && w.size(2) % 128 == 0,
      "W4 weights require [experts, rows, columns], columns divisible by 128");
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 &&
                  packed.size(0) == w.size(0) && packed.size(1) == w.size(1) &&
                  packed.size(2) == w.size(2) / 4 &&
                  scales.size(0) == w.size(0) && scales.size(1) == w.size(1) &&
                  scales.size(2) == w.size(2) / 128,
              "W4 packed weight or scale shape mismatch");
  const c10::cuda::CUDAGuard guard(w.device());
  int64_t groups = w.numel() / 128;
  if (!groups) return;
  pack_weights<<<(groups + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(w.data_ptr()), packed.data_ptr<uint16_t>(),
      reinterpret_cast<half*>(scales.data_ptr()), w.size(-1) / 128, groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void linear(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
            torch::Tensor ids, torch::Tensor out, bool gate, bool rounding) {
  check_tensor(x, torch::kHalf, x);
  check_tensor(packed, torch::kUInt16, x);
  check_tensor(scales, torch::kHalf, x);
  check_tensor(ids, torch::kInt32, x);
  check_tensor(out, torch::kHalf, x);
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && ids.dim() == 2 &&
                  packed.size(2) > 0 && packed.size(2) % 32 == 0 &&
                  ids.size(1) > 0 && ids.size(1) <= 32,
              "Invalid W4 dimensions or route count");
  TORCH_CHECK(scales.size(0) == packed.size(0) &&
                  scales.size(1) == packed.size(1) &&
                  scales.size(2) == packed.size(2) / 32,
              "W4 scale shape mismatch");
  TORCH_CHECK(!gate || packed.size(1) % 2 == 0, "W4 gate rows must be even");
  int64_t routes = ids.numel();
  int64_t expected_input_rows = gate ? ids.size(0) : routes;
  TORCH_CHECK(
      x.numel() == expected_input_rows * packed.size(2) * 4 &&
          out.numel() == routes * (gate ? packed.size(1) / 2 : packed.size(1)),
      "W4 input or output shape mismatch");
  const c10::cuda::CUDAGuard guard(x.device());
  int experts = packed.size(0), rows = packed.size(1);
  int columns = packed.size(2) * 4;
  int tokens = ids.size(0), topk = ids.size(1);
  int64_t count = int64_t(tokens) * topk * (gate ? rows / 2 : rows);
  if (!count) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (gate)
    routed_linear<true><<<(count + 3) / 4, 128, 0, stream>>>(
        reinterpret_cast<half*>(x.data_ptr()), packed.data_ptr<uint16_t>(),
        reinterpret_cast<half*>(scales.data_ptr()), ids.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr()), tokens, topk, rows, columns,
        experts, rounding);
  else
    routed_linear<false><<<(count + 3) / 4, 128, 0, stream>>>(
        reinterpret_cast<half*>(x.data_ptr()), packed.data_ptr<uint16_t>(),
        reinterpret_cast<half*>(scales.data_ptr()), ids.data_ptr<int>(),
        reinterpret_cast<half*>(out.data_ptr()), tokens, topk, rows, columns,
        experts, rounding);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void reduce_routes(const half* projected, const float* weights,
                              const int* ids, half* out, int tokens, int hidden,
                              int topk, int experts) {
  __shared__ int order[32];
  int token = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < topk) order[threadIdx.x] = -1;
  __syncthreads();
  if (threadIdx.x < topk) {
    int candidate = threadIdx.x;
    int id = ids[token * topk + candidate];
    if (id >= 0 && id < experts) {
      int rank = 0;
      for (int k = 0; k < topk; ++k) {
        int other = ids[token * topk + k];
        rank += other >= 0 && other < experts &&
                (other < id || (other == id && k < candidate));
      }
      order[rank] = candidate;
    }
  }
  __syncthreads();
  if (col >= hidden) return;
  float result = 0;
  for (int k = 0; k < topk; ++k) {
    int selected = order[k];
    if (selected < 0) break;
    int route = token * topk + selected;
    float value = __half2float(projected[int64_t(route) * hidden + col]);
    float weight = __half2float(__float2half_rn(weights[route]));
    float contribution = __half2float(__float2half_rn(value * weight));
    result = __half2float(__float2half_rn(result + contribution));
  }
  out[int64_t(token) * hidden + col] = __float2half_rn(result);
}

void combine(torch::Tensor projected, torch::Tensor weights, torch::Tensor ids,
             torch::Tensor out, int experts) {
  check_tensor(out, torch::kHalf, out);
  check_tensor(projected, torch::kHalf, out);
  check_tensor(weights, torch::kFloat, out);
  check_tensor(ids, torch::kInt32, out);
  TORCH_CHECK(out.dim() == 2 && ids.dim() == 2 && ids.size(0) == out.size(0) &&
                  ids.size(1) > 0 && ids.size(1) <= 32 &&
                  weights.sizes() == ids.sizes() &&
                  projected.numel() == ids.numel() * out.size(1) && experts > 0,
              "Invalid W4 reduction shapes or expert count");
  const c10::cuda::CUDAGuard guard(out.device());
  int64_t count = out.numel();
  if (!count) return;
  reduce_routes<<<dim3((out.size(1) + 255) / 256, out.size(0)), 256, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(projected.data_ptr()), weights.data_ptr<float>(),
      ids.data_ptr<int>(), reinterpret_cast<half*>(out.data_ptr()), out.size(0),
      out.size(1), ids.size(1), experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void routed_down_combine(const half* x, const uint16_t* packed,
                                    const half* scales, const int* ids,
                                    const float* weights, half* out, int rows,
                                    int columns, int experts) {
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows) return;
  int token = blockIdx.y;
  int expert = lane < 8 ? ids[token * 8 + lane] : -1;
  bool valid = expert >= 0 && expert < experts;
  int rank = 0;
  for (int k = 0; k < 8; ++k) {
    int other = __shfl_sync(0xffffffff, expert, k);
    rank += other >= 0 && other < experts &&
            (other < expert || (other == expert && k < lane));
  }
  int count = __popc(__ballot_sync(0xffffffff, valid));
  float result = 0;
  int groups = columns / 128;
  for (int selected = 0; selected < count; ++selected) {
    int candidate =
        __ffs(__ballot_sync(0xffffffff, valid && rank == selected)) - 1;
    int chosen = __shfl_sync(0xffffffff, expert, candidate);
    int route = token * 8 + candidate;
    const half* input = x + int64_t(route) * columns;
    int64_t base_row = int64_t(chosen) * rows + row;
    float value = 0;
    for (int group = 0; group < groups; ++group) {
      int64_t offset = (base_row * groups + group) * 32 + lane;
      uint16_t bits = packed[offset];
      half scale = scales[base_row * groups + group];
      half2 scale_pair = __halves2half2(scale, scale);
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float2 activation = __half22float2(*reinterpret_cast<const half2*>(
            input + group * 128 + lane * 4 + i));
        unsigned int code_bits = (bits >> (i * 4));
        unsigned int magic =
            0x64006400u | (code_bits & 15u) | ((code_bits & 240u) << 12);
        half2 codes = __hsub2(*reinterpret_cast<half2*>(&magic),
                              __float2half2_rn(1032.0f));
        float2 weight = __half22float2(__hmul2(codes, scale_pair));
        value = fmaf(activation.x, weight.x, value);
        value = fmaf(activation.y, weight.y, value);
      }
    }
    value = sum_warp(value);
    if (!lane) {
      float projected = __half2float(__float2half_rn(value));
      float weight = __half2float(__float2half_rn(weights[route]));
      float contribution = __half2float(__float2half_rn(projected * weight));
      result = __half2float(__float2half_rn(result + contribution));
    }
  }
  if (!lane) out[int64_t(token) * rows + row] = __float2half_rn(result);
}

void down_combine(torch::Tensor x, torch::Tensor packed, torch::Tensor scales,
                  torch::Tensor ids, torch::Tensor weights, torch::Tensor out) {
  check_tensor(x, torch::kHalf, x);
  check_tensor(packed, torch::kUInt16, x);
  check_tensor(scales, torch::kHalf, x);
  check_tensor(ids, torch::kInt32, x);
  check_tensor(weights, torch::kFloat, x);
  check_tensor(out, torch::kHalf, x);
  TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && ids.dim() == 2 &&
                  ids.size(1) == 8 && weights.sizes() == ids.sizes() &&
                  packed.size(0) > 0 && packed.size(2) > 0 &&
                  packed.size(2) % 32 == 0 &&
                  scales.size(0) == packed.size(0) &&
                  scales.size(1) == packed.size(1) &&
                  scales.size(2) == packed.size(2) / 32 && out.dim() == 2 &&
                  out.size(0) == ids.size(0) && out.size(1) == packed.size(1) &&
                  x.numel() == ids.numel() * packed.size(2) * 4,
              "Invalid fused W4 down/route shapes");
  uintptr_t input_begin = reinterpret_cast<uintptr_t>(x.data_ptr());
  uintptr_t output_begin = reinterpret_cast<uintptr_t>(out.data_ptr());
  TORCH_CHECK(input_begin + x.numel() * sizeof(half) <= output_begin ||
                  output_begin + out.numel() * sizeof(half) <= input_begin,
              "W4 fused down requires nonoverlapping input and output");
  const c10::cuda::CUDAGuard guard(x.device());
  if (!out.numel()) return;
  routed_down_combine<<<dim3((out.size(1) + 3) / 4, out.size(0)), 128, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(x.data_ptr()), packed.data_ptr<uint16_t>(),
      reinterpret_cast<half*>(scales.data_ptr()), ids.data_ptr<int>(),
      weights.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr()),
      out.size(1), packed.size(2) * 4, packed.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack);
  m.def("linear", &linear);
  m.def("combine", &combine);
  m.def("down_combine", &down_combine);
}
