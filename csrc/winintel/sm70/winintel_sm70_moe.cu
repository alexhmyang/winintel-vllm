// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
#include <cuda_fp16.h>
#include <cmath>
#include "winintel_sm70.h"

namespace winintel {
namespace sm70 {
namespace {

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ int local_expert(const MoeArgs& a, int route) {
  const int id = a.ids[route];
  if (id < 0 || id >= a.global_experts) return -1;
  const int local = a.expert_map ? a.expert_map[id] : id;
  return local >= 0 && local < a.experts ? local : -1;
}

__global__ void gate_up(MoeArgs a) {
  const int lane = threadIdx.x & 31;
  const int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (index >= int64_t(a.tokens) * a.topk * a.intermediate) return;
  const int route = index / a.intermediate;
  const int col = index % a.intermediate;
  const int expert = local_expert(a, route);
  float gate = 0, up = 0;
  if (expert >= 0) {
    const auto* x =
        static_cast<const half*>(a.input) + int64_t(route / a.topk) * a.hidden;
    const auto* w = static_cast<const half*>(a.w13) +
                    (int64_t(expert) * 2 * a.intermediate + col) * a.hidden;
    for (int h = lane; h < a.hidden; h += 32) {
      float value = __half2float(x[h]);
      if (a.weight_on_input)
        value = __half2float(__float2half_rn(value * a.weights[route]));
      gate = fmaf(value, __half2float(w[h]), gate);
      up = fmaf(value, __half2float(w[int64_t(a.intermediate) * a.hidden + h]),
                up);
    }
  }
  gate = warp_sum(gate);
  up = warp_sum(up);
  if (lane == 0) {
    gate = __half2float(__float2half_rn(gate));
    up = __half2float(__float2half_rn(up));
    float activated = gate / (1.0f + expf(-gate));
    if (a.hf_rounding)
      activated = __half2float(__float2half_rn(activated));
    static_cast<half*>(a.gated)[index] =
        __float2half_rn(activated * up);
  }
}

__global__ void down(MoeArgs a) {
  const int lane = threadIdx.x & 31;
  const int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (index >= int64_t(a.tokens) * a.topk * a.hidden) return;
  const int route = index / a.hidden;
  const int col = index % a.hidden;
  const int expert = local_expert(a, route);
  float value = 0;
  if (expert >= 0) {
    const auto* x =
        static_cast<const half*>(a.gated) + int64_t(route) * a.intermediate;
    const auto* w = static_cast<const half*>(a.w2) +
                    (int64_t(expert) * a.hidden + col) * a.intermediate;
    for (int i = lane; i < a.intermediate; i += 32)
      value = fmaf(__half2float(x[i]), __half2float(w[i]), value);
  }
  value = warp_sum(value);
  if (lane == 0)
    static_cast<half*>(a.projected)[index] = __float2half_rn(value);
}

__global__ void combine(MoeArgs a) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(a.tokens) * a.hidden) return;
  const int token = index / a.hidden;
  const int col = index % a.hidden;
  float result = 0;
  int previous_id = -1, previous_route = -1;
  for (int k = 0; k < a.topk; ++k) {
    int selected = k;
    if (a.hf_rounding) {
      selected = -1;
      for (int candidate = 0; candidate < a.topk; ++candidate) {
        const int route = token * a.topk + candidate;
        const int id = a.ids[route];
        if (local_expert(a, route) < 0 ||
            id < previous_id ||
            (id == previous_id && candidate <= previous_route))
          continue;
        if (selected < 0 || id < a.ids[token * a.topk + selected])
          selected = candidate;
      }
      if (selected < 0) break;
      previous_id = a.ids[token * a.topk + selected];
      previous_route = selected;
    }
    const int route = token * a.topk + selected;
    if (local_expert(a, route) < 0) continue;
    const float value = __half2float(
        static_cast<const half*>(a.projected)[int64_t(route) * a.hidden + col]);
    float weight = a.weight_on_input ? 1.0f : a.weights[route];
    if (a.hf_rounding) {
      weight = __half2float(__float2half_rn(weight));
      const float contribution = __half2float(__float2half_rn(value * weight));
      result = __half2float(__float2half_rn(result + contribution));
    } else {
      result += value * weight;
    }
  }
  static_cast<half*>(a.out)[index] = __float2half_rn(result);
}
}  // namespace

cudaError_t fused_moe(const MoeArgs& a, cudaStream_t stream) {
  if (a.tokens == 0) return cudaSuccess;
  gate_up<<<(int64_t(a.tokens) * a.topk * a.intermediate + 3) / 4, 128, 0,
            stream>>>(a);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  down<<<(int64_t(a.tokens) * a.topk * a.hidden + 3) / 4, 128, 0, stream>>>(a);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  combine<<<(int64_t(a.tokens) * a.hidden + 255) / 256, 256, 0, stream>>>(a);
  return cudaGetLastError();
}
}  // namespace sm70
}  // namespace winintel
