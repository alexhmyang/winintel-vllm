// SPDX-License-Identifier: Apache-2.0
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
// Four warps share one rounded decode partition.
template <int D>
__global__ void decode_parallel_scores(const PagedAttentionArgs args) {
  __shared__ float scores[D];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long work = blockIdx.x, outer = work / args.num_partitions;
  const int part = work % args.num_partitions;
  const int token = outer / args.num_heads, head = outer % args.num_heads;
  const int pos = args.token_pos[token];
  const int start = args.sliding_window > 0
      ? max(0, pos - args.sliding_window + 1) : 0;
  const int active = min(args.num_partitions, max(8, (pos - start + 128) / 128));
  const int count = (pos - start + active) / active;
  const int begin = start + part * count, end = min(pos, begin + count - 1);
  if (count > D) {
    if (!warp) paged_attention_work<D, false>(args, token, head, part);
    return;
  }
  if (part >= active || begin > end) {
    if (!tid) {
      args.partial_max[work] = -CUDART_INF_F;
      args.partial_sum[work] = 0;
    }
    for (int i = tid; i < D; i += 128) args.partial_acc[work * D + i] = 0;
    return;
  }
  const auto* query = static_cast<const __half*>(args.query) + outer * D;
  const auto* cache = static_cast<const __half*>(args.kv_cache);
  const long long stride = args.cache_block_stride > 0 ? args.cache_block_stride
      : static_cast<long long>(args.block_size) * args.num_kv_heads * 2 * D;
  const int kv_head = head / (args.num_heads / args.num_kv_heads);
  float q[D / 32];
  #pragma unroll
  for (int i = 0; i < D / 32; ++i) q[i] = __half2float(query[lane + i * 32]);
  for (int slot = warp; slot < count; slot += 4) {
    const int p = begin + slot;
    float score = -CUDART_INF_F;
    if (p <= end) {
      const int page = args.block_tables[
          static_cast<long long>(args.token_seq[token]) * args.max_blocks_per_seq +
          p / args.block_size];
      const long long offset = static_cast<long long>(page) * stride +
          (static_cast<long long>(p % args.block_size) * args.num_kv_heads + kv_head) *
          2 * D + lane;
      float dot = 0;
      #pragma unroll
      for (int i = 0; i < D / 32; ++i)
        dot += q[i] * load_cache(args, offset + i * 32, args.k_scale);
      score = to_scaled_score(warp_reduce_sum(dot), args.scale, args.softcap, true);
    }
    if (!lane) {
      scores[slot] = score;
      args.partial_acc[work * D + slot] = score;
    }
  }
  __syncthreads();
  if (!tid) {
    float maximum = -CUDART_INF_F, total = 0;
    for (int i = 0; i <= end - begin; ++i) {
      const float score = scores[i] * 1.4426950408889634f;
      const float next = fmaxf(maximum, score);
      total = total * (maximum == -CUDART_INF_F ? 0 : exp2f(maximum - next)) +
              exp2f(score - next);
      maximum = next;
    }
    args.partial_max[work] = maximum;
    args.partial_sum[work] = total;
  }
}

template <int D>
__global__ void decode_parallel_values(const PagedAttentionArgs args) {
  __shared__ float probabilities[D], accumulated[4 * D];
  __shared__ float maximum, denominator;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long work = blockIdx.x, outer = work / args.num_partitions;
  const int part = work % args.num_partitions;
  const int token = outer / args.num_heads, head = outer % args.num_heads;
  const int pos = args.token_pos[token];
  const int start = args.sliding_window > 0
      ? max(0, pos - args.sliding_window + 1) : 0;
  const int active = min(args.num_partitions, max(8, (pos - start + 128) / 128));
  const int count = (pos - start + active) / active;
  const int begin = start + part * count, end = min(pos, begin + count - 1);
  if (count > D) {
    if (!warp) paged_attention_hf_value_work<D, false>(args, work);
    return;
  }
  if (part >= active || begin > end) return;
  if (!tid) {
    maximum = -CUDART_INF_F;
    denominator = 0;
    for (int p = 0; p < active; ++p)
      maximum = fmaxf(maximum, args.partial_max[outer * args.num_partitions + p]);
    for (int p = 0; p < active; ++p) {
      const long long i = outer * args.num_partitions + p;
      if (args.partial_max[i] != -CUDART_INF_F)
        denominator += args.partial_sum[i] * exp2f(args.partial_max[i] - maximum);
    }
  }
  __syncthreads();
  for (int slot = tid; slot < count; slot += 128)
    probabilities[slot] = denominator > 0 && begin + slot <= end
        ? __half2float(__float2half_rn(exp2f(
            args.partial_acc[work * D + slot] * 1.4426950408889634f - maximum) /
            denominator)) : 0;
  __syncthreads();
  const auto* cache = static_cast<const __half*>(args.kv_cache);
  const long long stride = args.cache_block_stride > 0 ? args.cache_block_stride
      : static_cast<long long>(args.block_size) * args.num_kv_heads * 2 * D;
  const int kv_head = head / (args.num_heads / args.num_kv_heads);
  float acc[D / 32] = {};
  const int segment = (count + 3) / 4;
  for (int slot = warp * segment; slot < min(count, (warp + 1) * segment); ++slot) {
    const int p = begin + slot;
    if (p > end) break;
    const int page = args.block_tables[
        static_cast<long long>(args.token_seq[token]) * args.max_blocks_per_seq +
        p / args.block_size];
    const long long offset = static_cast<long long>(page) * stride +
        (static_cast<long long>(p % args.block_size) * args.num_kv_heads + kv_head) *
        2 * D + D + lane;
    const float weight = probabilities[slot];
    #pragma unroll
    for (int i = 0; i < D / 32; ++i)
      acc[i] += weight * load_cache(args, offset + i * 32, args.v_scale);
  }
  #pragma unroll
  for (int i = 0; i < D / 32; ++i)
    accumulated[warp * D + lane + i * 32] = acc[i];
  __syncthreads();
  for (int dim = tid; dim < D; dim += 128) {
    float value = accumulated[dim];
    #pragma unroll
    for (int w = 1; w < 4; ++w) value += accumulated[w * D + dim];
    args.partial_acc[work * D + dim] = value;
  }
}
