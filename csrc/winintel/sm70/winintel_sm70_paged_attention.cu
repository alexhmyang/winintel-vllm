// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
//
// Volta (SM70) paged attention kernels.
//
// Design notes
// ------------
// * One warp owns one (query token, query head, kv partition) work item.
//   Every lane holds head_size / 32 elements of Q, the accumulator and the
//   output, so the softmax reduction never leaves the warp and no shared
//   memory or block synchronization is required inside the KV loop.
// * Scores are kept in log2 space, which turns the online-softmax rescale
//   into a single exp2 instruction per KV slot.
// * Long contexts are split into `num_partitions` KV chunks that are merged
//   by a second, cheap reduction kernel.  This keeps one CTA from becoming
//   the serialization point when the batch is small (the usual decode case).
// * Decode uses plain FMA / half2float arithmetic. Experimental prefill uses
//   CUDA WMMA directly, without external attention or matrix libraries.

#include "winintel_sm70.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <mma.h>

namespace winintel {
namespace sm70 {
namespace {

constexpr unsigned kFullWarpMask = 0xffffffffu;
constexpr int kPartitionContext = 128;
constexpr int kMinDecodePartitions = 8;

// Reuse CUDA's software conversion used by vLLM's NVIDIA FP8 cache utilities.
// It supports SM70 without requiring native FP8 instructions.
__device__ __forceinline__ float load_cache(
    const PagedAttentionArgs& args, long long offset, float scale) {
  if (!args.fp8_cache) {
    return __half2float(static_cast<const __half*>(args.kv_cache)[offset]);
  }
  const auto bits = static_cast<const uint8_t*>(args.kv_cache)[offset];
  const __half value = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
  return __half2float(value) * scale;
}

__device__ __forceinline__ float to_log2_score(float score, float scale,
                                             float softcap,
                                             bool hf_rounding = false);

template <int HEAD_SIZE, bool ALLOW_SCORE_CACHE = false>
__device__ void paged_attention_work(const PagedAttentionArgs args, int token,
                                    int head, int partition);

// Four warps cooperate on sixteen queries against paged tiles. FP32 softmax and
// accumulation are retained; probabilities are rounded to FP16 for WMMA.
template <int D>
__global__ void tiled_prefill_kernel(const PagedAttentionArgs args) {
  using namespace nvcuda;
  __shared__ __align__(32) __half q[16 * D], k[16 * D], v[16 * D];
  __shared__ __align__(32) __half probabilities[256];
  __shared__ __align__(32) float scores[256], accumulated[16 * D];
  __shared__ float maxima[16], sums[16], rescale[16];
  __shared__ int positions[16], starts[16], tile_begin, tile_end;
  __shared__ int sequence;
  __shared__ bool mixed;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int head = blockIdx.y;
  const int first = blockIdx.x * 16;
  if (tid == 0) {
    sequence = args.token_seq[first];
    mixed = false;
    for (int row = 1; row < 16 && first + row < args.num_tokens; ++row) {
      mixed |= args.token_seq[first + row] != sequence;
    }
  }
  __syncthreads();
  if (mixed) {
    for (int row = warp; row < 16 && first + row < args.num_tokens; row += 4) {
      paged_attention_work<D>(args, first + row, head, 0);
    }
    return;
  }
  const int kv_head = head / (args.num_heads / args.num_kv_heads);
  const auto* query = static_cast<const __half*>(args.query);
  const auto* cache = static_cast<const __half*>(args.kv_cache);
  const long long stride = args.cache_block_stride > 0
      ? args.cache_block_stride
      : static_cast<long long>(args.block_size) * args.num_kv_heads * 2 * D;
  if (tid < 16) {
    const int token = first + tid;
    positions[tid] = token < args.num_tokens ? args.token_pos[token] : -1;
    starts[tid] = args.sliding_window > 0
        ? max(0, positions[tid] - args.sliding_window + 1) : 0;
    maxima[tid] = -CUDART_INF_F;
    sums[tid] = 0;
  }
  for (int i = tid; i < 16 * D; i += 128) {
    const int row = i / D, dim = i % D;
    q[i] = first + row < args.num_tokens
        ? query[(static_cast<long long>(first + row) * args.num_heads + head) * D + dim]
        : __float2half(0);
    accumulated[i] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    tile_begin = 0;
    tile_end = -1;
    int earliest = 0x7fffffff;
    for (int row = 0; row < 16; ++row) {
      if (positions[row] >= 0) earliest = min(earliest, starts[row]);
      tile_end = max(tile_end, positions[row]);
    }
    if (tile_end >= 0) tile_begin = earliest / 16 * 16;
  }
  __syncthreads();
  for (int base = tile_begin; base <= tile_end; base += 16) {
    for (int i = tid; i < 16 * D; i += 128) {
      const int pos = base + i / D, dim = i % D;
      __half key = __float2half(0), value = __float2half(0);
      if (pos <= tile_end) {
        const int page = args.block_tables[
            static_cast<long long>(sequence) * args.max_blocks_per_seq +
            pos / args.block_size];
        const long long offset = static_cast<long long>(page) * stride +
            (static_cast<long long>(pos % args.block_size) * args.num_kv_heads +
             kv_head) * 2 * D + dim;
        key = __float2half_rn(load_cache(args, offset, args.k_scale));
        value = __float2half_rn(load_cache(args, offset + D, args.v_scale));
      }
      k[i] = key;
      v[i] = value;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> product;
    if (warp == 0) {
      wmma::fill_fragment(product, 0.0f);
      for (int dim = 0; dim < D; dim += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, q + dim, D);
        wmma::load_matrix_sync(b, k + dim, D);
        wmma::mma_sync(product, a, b, product);
      }
      wmma::store_matrix_sync(scores, product, 16, wmma::mem_col_major);
    }
    __syncthreads();
    if (tid < 16) {
      float next_max = maxima[lane];
      for (int col = 0; col < 16; ++col) {
        const int pos = base + col;
        const bool valid = pos >= starts[lane] && pos <= positions[lane];
        scores[col * 16 + lane] = valid
            ? to_log2_score(scores[col * 16 + lane], args.scale, args.softcap)
            : -CUDART_INF_F;
        next_max = fmaxf(next_max, scores[col * 16 + lane]);
      }
      const float alpha = maxima[lane] == -CUDART_INF_F
          ? 0.0f : exp2f(maxima[lane] - next_max);
      float total = sums[lane] * alpha;
      for (int col = 0; col < 16; ++col) {
        const float score = scores[col * 16 + lane];
        const float weight = score == -CUDART_INF_F ? 0.0f : exp2f(score - next_max);
        probabilities[col * 16 + lane] = __float2half_rn(weight);
        total += weight;
      }
      rescale[lane] = alpha;
      sums[lane] = total;
      maxima[lane] = next_max;
    }
    __syncthreads();
    for (int i = tid; i < 16 * D; i += 128) {
      accumulated[i] *= rescale[i / D];
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> p;
    wmma::load_matrix_sync(p, probabilities, 16);
    for (int dim = warp * 16; dim < D; dim += 64) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> values;
      wmma::load_matrix_sync(values, v + dim, D);
      wmma::load_matrix_sync(product, accumulated + dim, D, wmma::mem_row_major);
      wmma::mma_sync(product, p, values, product);
      wmma::store_matrix_sync(accumulated + dim, product, D, wmma::mem_row_major);
    }
    __syncthreads();
  }
  auto* output = static_cast<__half*>(args.out);
  for (int i = tid; i < 16 * D; i += 128) {
    const int row = i / D, dim = i % D;
    if (first + row < args.num_tokens) {
      const float value = sums[row] > 0 ? accumulated[i] / sums[row] : 0;
      output[(static_cast<long long>(first + row) * args.num_heads + head) * D + dim] =
          __float2half_rn(value);
    }
  }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float to_scaled_score(float score, float scale,
                                               float softcap,
                                               bool hf_rounding) {
  if (hf_rounding) score = __half2float(__float2half_rn(score));
  float s = score * scale;
  if (hf_rounding) s = __half2float(__float2half_rn(s));
  if (softcap > 0.0f) {
    s = softcap * tanhf(s / softcap);
  }
  return s;
}

__device__ __forceinline__ float to_log2_score(float score, float scale,
                                             float softcap,
                                             bool hf_rounding) {
  return to_scaled_score(score, scale, softcap, hf_rounding) *
         1.4426950408889634f;
}

template <int HEAD_SIZE, bool ALLOW_SCORE_CACHE>
__device__ void paged_attention_work(const PagedAttentionArgs args, int token,
                                    int head, int partition) {
  constexpr int kVec = HEAD_SIZE / 32;
  constexpr int kHalfSize = 2 * HEAD_SIZE;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const long long outer = static_cast<long long>(token) * args.num_heads + head;

  const int seq = args.token_seq[token];
  const int pos = args.token_pos[token];
  const int start =
      args.sliding_window > 0 ? max(0, pos - args.sliding_window + 1) : 0;
  const int length = pos - start + 1;

  // Extra graph launch partitions stay empty instead of changing KV reductions.
  const int active_partitions =
      min(args.num_partitions,
          max(kMinDecodePartitions,
              (length + kPartitionContext - 1) / kPartitionContext));
  const int per_partition =
      (length + active_partitions - 1) / active_partitions;
  const bool cache_scores = ALLOW_SCORE_CACHE && per_partition <= HEAD_SIZE;
  const int kv_begin = start + partition * per_partition;
  const int kv_end = min(pos, kv_begin + per_partition - 1);
  const long long index = outer * args.num_partitions + partition;

  if (partition >= active_partitions) {
    if (lane == 0) {
      args.partial_max[index] = -CUDART_INF_F;
      args.partial_sum[index] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      args.partial_acc[index * HEAD_SIZE + lane + i * 32] = 0.0f;
    }
    return;
  }

  const __half* q =
      reinterpret_cast<const __half*>(args.query) +
      (static_cast<long long>(token) * args.num_heads + head) * HEAD_SIZE;
  const __half* kv = reinterpret_cast<const __half*>(args.kv_cache);
  const long long block_stride = args.cache_block_stride > 0
                                     ? args.cache_block_stride
                                     : static_cast<long long>(args.block_size) *
                                           args.num_kv_heads * kHalfSize;

  float q_reg[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    q_reg[i] = __half2float(q[lane + i * 32]);
  }

  float acc[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    acc[i] = 0.0f;
  }
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;

  if (kv_begin <= kv_end) {
    const int group_size = args.num_heads / args.num_kv_heads;
    const int kv_head = head / group_size;
    const int block_size = args.block_size;
    const int first_block = kv_begin / block_size;
    const int last_block = kv_end / block_size;

    for (int block = first_block; block <= last_block; ++block) {
      const int physical_block = args.block_tables[static_cast<long long>(seq) *
                                                       args.max_blocks_per_seq +
                                                   block];
      const int slot_begin =
          block == first_block ? kv_begin - block * block_size : 0;
      const int slot_end =
          block == last_block ? kv_end - block * block_size : block_size - 1;

      for (int slot = slot_begin; slot <= slot_end; ++slot) {
        const long long base =
            static_cast<long long>(physical_block) * block_stride +
            (static_cast<long long>(slot) * args.num_kv_heads + kv_head) *
                kHalfSize +
            lane;

        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          dot += q_reg[i] * load_cache(args, base + i * 32, args.k_scale);
        }
        dot = warp_reduce_sum(dot);

        const float score =
            to_log2_score(dot, args.scale, args.softcap, args.hf_rounding);
        if (cache_scores && lane == 0) {
          args.partial_acc[index * HEAD_SIZE +
                           block * block_size + slot - kv_begin] =
              to_scaled_score(dot, args.scale, args.softcap, true);
        }
        const float new_max = fmaxf(running_max, score);
        const float alpha =
            running_max == -CUDART_INF_F ? 0.0f : exp2f(running_max - new_max);
        const float weight = exp2f(score - new_max);
        running_sum = running_sum * alpha + weight;
        running_max = new_max;

        if (!args.hf_rounding) {
#pragma unroll
          for (int i = 0; i < kVec; ++i) {
            acc[i] = acc[i] * alpha +
                     weight * load_cache(args, base + HEAD_SIZE + i * 32,
                                         args.v_scale);
          }
        }
      }
    }
  }

  if (args.num_partitions == 1) {
    __half* out =
        reinterpret_cast<__half*>(args.out) +
        (static_cast<long long>(token) * args.num_heads + head) * HEAD_SIZE;
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      out[lane + i * 32] = __float2half_rn(acc[i] * inv);
    }
    return;
  }

  if (lane == 0) {
    args.partial_max[index] = running_max;
    args.partial_sum[index] = running_sum;
  }
  // Rounded attention reuses this workspace for scores until the value pass.
  float* partial = args.partial_acc + index * HEAD_SIZE;
  if (cache_scores) return;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    partial[lane + i * 32] = acc[i];
  }
}

template <int HEAD_SIZE, int WARPS_PER_BLOCK, bool ALLOW_SCORE_CACHE>
__global__ void __launch_bounds__(32 * WARPS_PER_BLOCK)
    paged_attention_partition_kernel(const PagedAttentionArgs args) {
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const long long work =
      static_cast<long long>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  const long long total = static_cast<long long>(args.num_tokens) *
                          args.num_heads * args.num_partitions;
  if (work >= total) return;
  const int partition = static_cast<int>(work % args.num_partitions);
  const long long outer = work / args.num_partitions;
  paged_attention_work<HEAD_SIZE, ALLOW_SCORE_CACHE>(
      args, static_cast<int>(outer / args.num_heads),
      static_cast<int>(outer % args.num_heads), partition);
}

template <int HEAD_SIZE, bool ALLOW_SCORE_CACHE>
__device__ void paged_attention_hf_value_work(
    const PagedAttentionArgs args, long long work) {
  constexpr int kVec = HEAD_SIZE / 32;
  const int lane = threadIdx.x & 31;
  if (work >= static_cast<long long>(args.num_tokens) * args.num_heads *
                  args.num_partitions) return;
  const int partition = work % args.num_partitions;
  const long long outer = work / args.num_partitions;
  const int token = outer / args.num_heads, head = outer % args.num_heads;
  const int pos = args.token_pos[token];
  const int start =
      args.sliding_window > 0 ? max(0, pos - args.sliding_window + 1) : 0;
  const int active = min(
      args.num_partitions,
      max(kMinDecodePartitions,
          (pos - start + 1 + kPartitionContext - 1) / kPartitionContext));
  const long long partial = outer * args.num_partitions;
  if (partition >= active) return;
  float maximum = -CUDART_INF_F, denominator = 0;
  for (int p = 0; p < active; ++p)
    maximum = fmaxf(maximum, args.partial_max[partial + p]);
  for (int p = 0; p < active; ++p) {
    const float part_max = args.partial_max[partial + p];
    if (part_max != -CUDART_INF_F)
      denominator += args.partial_sum[partial + p] * exp2f(part_max - maximum);
  }
  const auto* query = static_cast<const __half*>(args.query) + outer * HEAD_SIZE;
  const auto* cache = static_cast<const __half*>(args.kv_cache);
  const long long stride = args.cache_block_stride > 0
      ? args.cache_block_stride
      : static_cast<long long>(args.block_size) * args.num_kv_heads * 2 * HEAD_SIZE;
  const int sequence = args.token_seq[token];
  const int kv_head = head / (args.num_heads / args.num_kv_heads);
  const int per_partition = (pos - start + 1 + active - 1) / active;
  const int begin = start + partition * per_partition;
  const int end = min(pos, begin + per_partition - 1);
  const bool cached = ALLOW_SCORE_CACHE && per_partition <= HEAD_SIZE;
  float q[kVec], acc[kVec], probabilities[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    q[i] = cached ? 0 : __half2float(query[lane + i * 32]);
    acc[i] = 0;
    probabilities[i] = 0;
    if (cached && begin + lane + i * 32 <= end && denominator > 0) {
      const float score = args.partial_acc[work * HEAD_SIZE + lane + i * 32] *
                          1.4426950408889634f;
      probabilities[i] = __half2float(__float2half_rn(
          exp2f(score - maximum) / denominator));
    }
  }
  for (int position = begin; position <= end && denominator > 0; ++position) {
    const int page = args.block_tables[
        static_cast<long long>(sequence) * args.max_blocks_per_seq +
        position / args.block_size];
    const long long offset = static_cast<long long>(page) * stride +
        (static_cast<long long>(position % args.block_size) * args.num_kv_heads +
         kv_head) * 2 * HEAD_SIZE + lane;
    float probability;
    if (cached) {
      const int slot = position - begin;
      float selected = 0;
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        if (slot / 32 == i) selected = probabilities[i];
      probability = __shfl_sync(kFullWarpMask, selected, slot & 31);
    } else {
      float dot = 0;
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        dot += q[i] * load_cache(args, offset + i * 32, args.k_scale);
      dot = warp_reduce_sum(dot);
      const float score = to_log2_score(dot, args.scale, args.softcap, true);
      probability = __half2float(__float2half_rn(
          exp2f(score - maximum) / denominator));
    }
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      acc[i] += probability *
          load_cache(args, offset + HEAD_SIZE + i * 32, args.v_scale);
  }
  float* output = args.partial_acc + work * HEAD_SIZE;
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    output[lane + i * 32] = acc[i];
}

template <int HEAD_SIZE, int WARPS_PER_BLOCK, bool ALLOW_SCORE_CACHE>
__global__ void paged_attention_hf_value_kernel(const PagedAttentionArgs args) {
  const long long work = static_cast<long long>(blockIdx.x) * WARPS_PER_BLOCK +
                         (threadIdx.x >> 5);
  paged_attention_hf_value_work<HEAD_SIZE, ALLOW_SCORE_CACHE>(args, work);
}

template <int HEAD_SIZE, int WARPS_PER_BLOCK>
__global__ void paged_attention_hf_merge_kernel(const PagedAttentionArgs args) {
  constexpr int kVec = HEAD_SIZE / 32;
  const int lane = threadIdx.x & 31;
  const long long work =
      static_cast<long long>(blockIdx.x) * WARPS_PER_BLOCK + (threadIdx.x >> 5);
  if (work >= static_cast<long long>(args.num_tokens) * args.num_heads) return;
  const int pos = args.token_pos[work / args.num_heads];
  const int start =
      args.sliding_window > 0 ? max(0, pos - args.sliding_window + 1) : 0;
  const int active = min(
      args.num_partitions,
      max(kMinDecodePartitions,
          (pos - start + 1 + kPartitionContext - 1) / kPartitionContext));
  float acc[kVec] = {};
  for (int p = 0; p < active; ++p) {
    const float* partial =
        args.partial_acc + (work * args.num_partitions + p) * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < kVec; ++i)
      acc[i] += partial[lane + i * 32];
  }
  auto* output = static_cast<__half*>(args.out) + work * HEAD_SIZE;
#pragma unroll
  for (int i = 0; i < kVec; ++i)
    output[lane + i * 32] = __float2half_rn(acc[i]);
}

template <int HEAD_SIZE, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(32 * WARPS_PER_BLOCK)
    paged_attention_merge_kernel(const PagedAttentionArgs args) {
  constexpr int kVec = HEAD_SIZE / 32;

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const long long work =
      static_cast<long long>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  const long long total =
      static_cast<long long>(args.num_tokens) * args.num_heads;
  if (work >= total) {
    return;
  }

  const int head = static_cast<int>(work % args.num_heads);
  const int token = static_cast<int>(work / args.num_heads);
  const long long base =
      (static_cast<long long>(token) * args.num_heads + head) *
      args.num_partitions;
  const int pos = args.token_pos[token];
  const int start =
      args.sliding_window > 0 ? max(0, pos - args.sliding_window + 1) : 0;
  const int length = pos - start + 1;
  const int active_partitions =
      min(args.num_partitions,
          max(kMinDecodePartitions,
              (length + kPartitionContext - 1) / kPartitionContext));

  float global_max = -CUDART_INF_F;
  for (int p = 0; p < active_partitions; ++p) {
    global_max = fmaxf(global_max, args.partial_max[base + p]);
  }

  float denominator = 0.0f;
  float acc[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    acc[i] = 0.0f;
  }

  for (int p = 0; p < active_partitions; ++p) {
    const float part_max = args.partial_max[base + p];
    const float weight =
        part_max == -CUDART_INF_F ? 0.0f : exp2f(part_max - global_max);
    denominator += args.partial_sum[base + p] * weight;
    const float* partial = args.partial_acc + (base + p) * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      acc[i] += partial[lane + i * 32] * weight;
    }
  }

  __half* out =
      reinterpret_cast<__half*>(args.out) +
      (static_cast<long long>(token) * args.num_heads + head) * HEAD_SIZE;
  const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    out[lane + i * 32] = __float2half_rn(acc[i] * inv);
  }
}

__global__ void reshape_and_cache_kernel(const CacheUpdateArgs args) {
  const long long index =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(args.num_tokens) *
                          args.num_kv_heads * args.head_size;
  if (index >= total) {
    return;
  }

  const int dim = static_cast<int>(index % args.head_size);
  const int kv_head =
      static_cast<int>((index / args.head_size) % args.num_kv_heads);
  const int token = static_cast<int>(
      index / (static_cast<long long>(args.head_size) * args.num_kv_heads));

  const int64_t slot = args.slot_mapping[token];
  if (slot < 0) {
    return;
  }

  const int block = static_cast<int>(slot / args.block_size);
  const int offset = static_cast<int>(slot % args.block_size);
  const long long block_stride =
      args.cache_block_stride > 0 ? args.cache_block_stride
                                  : static_cast<long long>(args.block_size) *
                                        args.num_kv_heads * 2 * args.head_size;
  const long long dst =
      static_cast<long long>(block) * block_stride +
      (static_cast<long long>(offset) * args.num_kv_heads + kv_head) * 2 *
          args.head_size +
      dim;

  const long long src =
      (static_cast<long long>(token) * args.num_kv_heads + kv_head) *
          args.head_size +
      dim;

  __half* cache = reinterpret_cast<__half*>(args.kv_cache);
  const __half* key = reinterpret_cast<const __half*>(args.key);
  const __half* value = reinterpret_cast<const __half*>(args.value);

  if (args.fp8_cache) {
    auto* packed = static_cast<uint8_t*>(args.kv_cache);
    packed[dst] = __nv_cvt_float_to_fp8(
        __half2float(key[src]) / args.k_scale, __NV_SATFINITE, __NV_E4M3);
    packed[dst + args.head_size] = __nv_cvt_float_to_fp8(
        __half2float(value[src]) / args.v_scale, __NV_SATFINITE, __NV_E4M3);
    return;
  }
  cache[dst] = __float2half_rn(__half2float(key[src]) * args.k_scale);
  cache[dst + args.head_size] =
      __float2half_rn(__half2float(value[src]) * args.v_scale);
}

#include "winintel_sm70_cooperative_decode.cuh"

template <int HEAD_SIZE>
cudaError_t launch_paged_attention(const PagedAttentionArgs& args,
                                   cudaStream_t stream) {
  constexpr int kWarpPerBlock = 4;
  const long long units = static_cast<long long>(args.num_tokens) *
                          args.num_heads * args.num_partitions;
  const int blocks =
      static_cast<int>((units + kWarpPerBlock - 1) / kWarpPerBlock);

  const bool cache_scores = args.hf_rounding &&
      static_cast<long long>(args.max_blocks_per_seq) * args.block_size <=
          static_cast<long long>(HEAD_SIZE) * args.num_partitions;
  if constexpr (HEAD_SIZE == 128) {
    if (args.cooperative_decode && args.hf_rounding && args.num_tokens <= 3 &&
        args.num_partitions > 1 &&
        static_cast<long long>(args.max_blocks_per_seq) * args.block_size <= 4096) {
      decode_parallel_scores<HEAD_SIZE><<<units, 128, 0, stream>>>(args);
      auto error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      decode_parallel_values<HEAD_SIZE><<<units, 128, 0, stream>>>(args);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      const long long heads =
          static_cast<long long>(args.num_tokens) * args.num_heads;
      paged_attention_hf_merge_kernel<HEAD_SIZE, 4>
          <<<(heads + 3) / 4, 128, 0, stream>>>(args);
      return cudaGetLastError();
    }
  }
  if (cache_scores) {
    paged_attention_partition_kernel<HEAD_SIZE, kWarpPerBlock, true>
        <<<blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
  } else {
    paged_attention_partition_kernel<HEAD_SIZE, kWarpPerBlock, false>
        <<<blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
  }
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  if (args.num_partitions > 1) {
    const long long merge_units =
        static_cast<long long>(args.num_tokens) * args.num_heads;
    const int merge_blocks =
        static_cast<int>((merge_units + kWarpPerBlock - 1) / kWarpPerBlock);
    if (args.hf_rounding) {
      if (cache_scores) {
        paged_attention_hf_value_kernel<HEAD_SIZE, kWarpPerBlock, true>
            <<<blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
      } else {
        paged_attention_hf_value_kernel<HEAD_SIZE, kWarpPerBlock, false>
            <<<blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
      }
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
      paged_attention_hf_merge_kernel<HEAD_SIZE, kWarpPerBlock>
          <<<merge_blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
    } else {
      paged_attention_merge_kernel<HEAD_SIZE, kWarpPerBlock>
          <<<merge_blocks, 32 * kWarpPerBlock, 0, stream>>>(args);
    }
    error = cudaGetLastError();
  }
  return error;
}

}  // namespace

cudaError_t paged_attention(const PagedAttentionArgs& args,
                            cudaStream_t stream) {
  if (args.num_partitions < 1) {
    return cudaErrorInvalidValue;
  }
  if (args.hf_rounding && (args.num_partitions < 2 || args.tiled_prefill))
    return cudaErrorInvalidValue;
  if (args.num_tokens == 0) {
    return cudaSuccess;
  }
  if (args.tiled_prefill) {
    if (args.num_partitions != 1) return cudaErrorInvalidValue;
    const dim3 grid((args.num_tokens + 15) / 16, args.num_heads);
    if (args.head_size == 64) {
      tiled_prefill_kernel<64><<<grid, 128, 0, stream>>>(args);
    } else if (args.head_size == 128) {
      tiled_prefill_kernel<128><<<grid, 128, 0, stream>>>(args);
    } else {
      return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
  }
  switch (args.head_size) {
    case 64:
      return launch_paged_attention<64>(args, stream);
    case 128:
      return launch_paged_attention<128>(args, stream);
    case 256:
      return launch_paged_attention<256>(args, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t reshape_and_cache(const CacheUpdateArgs& args,
                              cudaStream_t stream) {
  const long long total = static_cast<long long>(args.num_tokens) *
                          args.num_kv_heads * args.head_size;
  if (total == 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  reshape_and_cache_kernel<<<blocks, threads, 0, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace sm70
}  // namespace winintel
