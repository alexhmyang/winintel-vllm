// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>

#include "winintel_sm70.h"

void winintel_gathered_qk_norm(torch::Tensor input, torch::Tensor qw,
                               torch::Tensor kw, torch::Tensor qout,
                               torch::Tensor kout, int64_t rank,
                               double epsilon);

namespace {

void check_input(const torch::Tensor& tensor, const char* name, int dims) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.dim() == dims, name, " must have ", dims, " dimensions");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_cache(const torch::Tensor& cache) {
  TORCH_CHECK(cache.is_cuda(), "kv_cache must be a CUDA tensor");
  TORCH_CHECK(cache.dim() == 4, "kv_cache must have 4 dimensions");
  TORCH_CHECK(cache.stride(3) == 1 && cache.stride(2) == cache.size(3) &&
                  cache.stride(1) == cache.size(2) * cache.size(3) &&
                  cache.stride(0) >= cache.size(1) * cache.stride(1),
              "kv_cache must have contiguous pages with optional padding "
              "between blocks");
}

void check_device(const torch::Tensor& tensor, const torch::Tensor& input,
                  const char* name) {
  TORCH_CHECK(tensor.device() == input.device(), name,
              " must be on the same device as the input");
}

void paged_attention(torch::Tensor out, const torch::Tensor& query,
                     const torch::Tensor& kv_cache,
                     const torch::Tensor& block_tables,
                     const torch::Tensor& token_seq,
                     const torch::Tensor& token_pos, torch::Tensor partial_acc,
                     torch::Tensor partial_max, torch::Tensor partial_sum,
                     double scale, int64_t block_size, int64_t num_kv_heads,
                     int64_t num_partitions, double softcap,
                     int64_t sliding_window, bool tiled_prefill,
                     bool hf_rounding, bool cooperative_decode,
                     double k_scale, double v_scale) {
  check_input(out, "out", 3);
  check_input(query, "query", 3);
  check_cache(kv_cache);
  check_input(block_tables, "block_tables", 2);
  check_input(token_seq, "token_seq", 1);
  check_input(token_pos, "token_pos", 1);

  TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be float16");
  TORCH_CHECK(query.scalar_type() == torch::kHalf, "query must be float16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kHalf ||
                  kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be float16 or uint8 E4M3");
  TORCH_CHECK(k_scale > 0 && v_scale > 0 &&
                  std::isfinite(k_scale) && std::isfinite(v_scale),
              "KV scales must be finite and positive");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32,
              "block_tables must be int32");
  TORCH_CHECK(token_seq.scalar_type() == torch::kInt32,
              "token_seq must be int32");
  TORCH_CHECK(token_pos.scalar_type() == torch::kInt32,
              "token_pos must be int32");

  const int64_t num_tokens = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_size = query.size(2);
  TORCH_CHECK(num_heads > 0 && num_kv_heads > 0 && block_size > 0,
              "head counts and block_size must be positive");
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "unsupported head_size ", head_size);
  TORCH_CHECK(out.sizes() == query.sizes(), "out shape must match query");
  TORCH_CHECK(kv_cache.size(1) == block_size,
              "block_size does not match cache");
  TORCH_CHECK(kv_cache.size(2) == num_kv_heads,
              "num_kv_heads does not match cache");
  TORCH_CHECK(kv_cache.size(3) == 2 * head_size,
              "kv_cache head dimension does not match query");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");
  TORCH_CHECK(num_partitions >= 1, "num_partitions must be positive");
  TORCH_CHECK(
      token_seq.size(0) == num_tokens && token_pos.size(0) == num_tokens,
      "token_seq/token_pos lengths must match query");
  check_device(out, query, "out");
  check_device(kv_cache, query, "kv_cache");
  check_device(block_tables, query, "block_tables");
  check_device(token_seq, query, "token_seq");
  check_device(token_pos, query, "token_pos");
  const c10::cuda::CUDAGuard guard(query.device());

  winintel::sm70::PagedAttentionArgs args{};
  args.query = query.data_ptr();
  args.kv_cache = kv_cache.data_ptr();
  args.out = out.data_ptr();
  args.block_tables = block_tables.data_ptr<int>();
  args.token_seq = token_seq.data_ptr<int>();
  args.token_pos = token_pos.data_ptr<int>();
  args.partial_acc = nullptr;
  args.partial_max = nullptr;
  args.partial_sum = nullptr;
  if (num_partitions > 1) {
    check_input(partial_acc, "partial_acc", 4);
    check_input(partial_max, "partial_max", 3);
    check_input(partial_sum, "partial_sum", 3);
    const std::vector<int64_t> partial_shape{num_tokens, num_heads,
                                             num_partitions};
    auto acc_shape = partial_shape;
    acc_shape.push_back(head_size);
    TORCH_CHECK(partial_acc.sizes() == torch::IntArrayRef(acc_shape) &&
                    partial_max.sizes() == torch::IntArrayRef(partial_shape) &&
                    partial_sum.sizes() == torch::IntArrayRef(partial_shape),
                "partition workspace shapes do not match query");
    check_device(partial_acc, query, "partial_acc");
    check_device(partial_max, query, "partial_max");
    check_device(partial_sum, query, "partial_sum");
    args.partial_acc = partial_acc.data_ptr<float>();
    args.partial_max = partial_max.data_ptr<float>();
    args.partial_sum = partial_sum.data_ptr<float>();
  }
  args.num_tokens = static_cast<int>(num_tokens);
  args.num_heads = static_cast<int>(num_heads);
  args.num_kv_heads = static_cast<int>(num_kv_heads);
  args.head_size = static_cast<int>(head_size);
  args.block_size = static_cast<int>(block_size);
  args.max_blocks_per_seq = static_cast<int>(block_tables.size(1));
  args.num_partitions = static_cast<int>(num_partitions);
  args.scale = static_cast<float>(scale);
  args.softcap = static_cast<float>(softcap);
  args.sliding_window = static_cast<int>(sliding_window);
  args.cache_block_stride = kv_cache.stride(0);
  TORCH_CHECK(!tiled_prefill || (num_partitions == 1 && head_size <= 128),
              "tiled prefill requires one partition and "
              "head_size 64 or 128");
  args.tiled_prefill = tiled_prefill;
  TORCH_CHECK(
      !hf_rounding || (num_partitions > 1 && !tiled_prefill),
      "reference-style rounding requires partitions and scalar prefill");
  args.hf_rounding = hf_rounding;
  args.cooperative_decode = cooperative_decode;
  args.fp8_cache = kv_cache.scalar_type() == torch::kUInt8;
  args.k_scale = static_cast<float>(k_scale);
  args.v_scale = static_cast<float>(v_scale);

  const cudaError_t error =
      winintel::sm70::paged_attention(args, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(error == cudaSuccess,
              "winintel paged_attention failed: ", cudaGetErrorString(error));
}

void reshape_and_cache(torch::Tensor key, torch::Tensor value,
                       torch::Tensor kv_cache, torch::Tensor slot_mapping,
                       double k_scale, double v_scale) {
  check_input(key, "key", 3);
  check_input(value, "value", 3);
  check_cache(kv_cache);
  check_input(slot_mapping, "slot_mapping", 1);
  TORCH_CHECK(key.scalar_type() == torch::kHalf, "key must be float16");
  TORCH_CHECK(value.scalar_type() == torch::kHalf, "value must be float16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kHalf ||
                  kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be float16 or uint8 E4M3");
  TORCH_CHECK(k_scale > 0 && v_scale > 0 &&
                  std::isfinite(k_scale) && std::isfinite(v_scale),
              "KV scales must be finite and positive");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
  TORCH_CHECK(slot_mapping.size(0) == key.size(0),
              "slot_mapping length mismatch");
  TORCH_CHECK(kv_cache.size(2) == key.size(1),
              "num_kv_heads does not match cache");
  TORCH_CHECK(kv_cache.size(3) == 2 * key.size(2),
              "kv_cache head dimension does not match key");
  TORCH_CHECK(kv_cache.size(1) > 0 && key.size(1) > 0 && key.size(2) > 0,
              "block_size and head dimensions must be positive");
  check_device(value, key, "value");
  check_device(kv_cache, key, "kv_cache");
  check_device(slot_mapping, key, "slot_mapping");
  const c10::cuda::CUDAGuard guard(key.device());

  winintel::sm70::CacheUpdateArgs args{};
  args.key = key.data_ptr();
  args.value = value.data_ptr();
  args.kv_cache = kv_cache.data_ptr();
  args.slot_mapping = slot_mapping.data_ptr<int64_t>();
  args.num_tokens = static_cast<int>(key.size(0));
  args.num_kv_heads = static_cast<int>(key.size(1));
  args.head_size = static_cast<int>(key.size(2));
  args.block_size = static_cast<int>(kv_cache.size(1));
  args.k_scale = static_cast<float>(k_scale);
  args.v_scale = static_cast<float>(v_scale);
  args.cache_block_stride = kv_cache.stride(0);
  args.fp8_cache = kv_cache.scalar_type() == torch::kUInt8;

  const cudaError_t error =
      winintel::sm70::reshape_and_cache(args, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(error == cudaSuccess,
              "winintel reshape_and_cache failed: ", cudaGetErrorString(error));
}

void fused_moe(torch::Tensor out, const torch::Tensor& input,
               const torch::Tensor& w13, const torch::Tensor& w2,
               const torch::Tensor& weights, const torch::Tensor& ids,
               const c10::optional<torch::Tensor>& expert_map,
               torch::Tensor gated, torch::Tensor projected,
               bool weight_on_input, bool hf_rounding) {
  check_input(input, "input", 2);
  check_input(w13, "w13", 3);
  check_input(w2, "w2", 3);
  check_input(out, "out", 2);
  check_input(weights, "weights", 2);
  check_input(ids, "ids", 2);
  check_input(gated, "gated", 3);
  check_input(projected, "projected", 3);
  for (const auto& t : {out, w13, w2, gated, projected}) {
    check_device(t, input, "expert tensor");
    TORCH_CHECK(t.scalar_type() == torch::kHalf,
                "expert tensors must be float16");
  }
  TORCH_CHECK(input.scalar_type() == torch::kHalf, "input must be float16");
  check_device(weights, input, "weights");
  check_device(ids, input, "ids");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat,
              "weights must be float32");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32, "ids must be int32");
  const int64_t m = input.size(0), h = input.size(1);
  const int64_t e = w13.size(0), i = w2.size(2), k = ids.size(1);
  TORCH_CHECK(h > 0 && e > 0 && i > 0 && k > 0,
              "MoE dimensions must be positive");
  const auto max_dim = std::numeric_limits<int>::max();
  TORCH_CHECK(h <= max_dim && e <= max_dim && i <= max_dim && k <= max_dim &&
                  m <= max_dim / k && (m * k * i + 3) / 4 <= max_dim &&
                  (m * k * h + 3) / 4 <= max_dim,
              "MoE dimensions exceed native kernel indexing limits");
  TORCH_CHECK(w13.size(1) == 2 * i && w13.size(2) == h && w2.size(0) == e &&
                  w2.size(1) == h,
              "MoE weight shape mismatch");
  TORCH_CHECK(ids.size(0) == m && weights.sizes() == ids.sizes() &&
                  out.sizes() == input.sizes(),
              "MoE routing/output shape mismatch");
  TORCH_CHECK(gated.size(0) == m && gated.size(1) == k && gated.size(2) == i &&
                  projected.size(0) == m && projected.size(1) == k &&
                  projected.size(2) == h,
              "MoE workspace shape mismatch");
  int64_t global_experts = e;
  if (expert_map) {
    check_input(*expert_map, "expert_map", 1);
    check_device(*expert_map, input, "expert_map");
    TORCH_CHECK(expert_map->scalar_type() == torch::kInt32,
                "expert_map must be int32");
    global_experts = expert_map->numel();
  }
  TORCH_CHECK(global_experts <= max_dim, "expert_map exceeds indexing limits");
  const c10::cuda::CUDAGuard guard(input.device());
  winintel::sm70::MoeArgs a{};
  a.input = input.data_ptr();
  a.w13 = w13.data_ptr();
  a.w2 = w2.data_ptr();
  a.weights = weights.data_ptr<float>();
  a.ids = ids.data_ptr<int>();
  a.expert_map = expert_map ? expert_map->data_ptr<int>() : nullptr;
  a.gated = gated.data_ptr();
  a.projected = projected.data_ptr();
  a.out = out.data_ptr();
  a.tokens = m;
  a.hidden = h;
  a.intermediate = i;
  a.experts = e;
  a.global_experts = global_experts;
  a.topk = k;
  a.weight_on_input = weight_on_input;
  a.hf_rounding = hf_rounding;
  const auto error =
      winintel::sm70::fused_moe(a, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(error == cudaSuccess,
              "winintel fused_moe failed: ", cudaGetErrorString(error));
}

}  // namespace

TORCH_LIBRARY(winintel_sm70, m) {
  m.def(
      "gathered_qk_norm(Tensor input, Tensor qw, Tensor kw, "
      "Tensor! qout, Tensor! kout, int rank, float epsilon) -> ()");
  m.def(
      "fused_moe(Tensor! out, Tensor input, Tensor w13, Tensor w2, "
      "Tensor weights, Tensor ids, Tensor? expert_map, Tensor! gated, "
      "Tensor! projected, bool weight_on_input, bool hf_rounding=False) -> ()");
  m.def(
      "paged_attention(Tensor! out, Tensor query, Tensor kv_cache, "
      "Tensor block_tables, Tensor token_seq, Tensor token_pos, "
      "Tensor! partial_acc, Tensor! partial_max, Tensor! partial_sum, "
      "float scale, int block_size, int num_kv_heads, int num_partitions, "
      "float softcap, int sliding_window, bool tiled_prefill=False, "
      "bool hf_rounding=False, bool cooperative_decode=False, "
      "float k_scale=1.0, float v_scale=1.0) -> ()");
  m.def(
      "reshape_and_cache(Tensor key, Tensor value, Tensor! kv_cache, "
      "Tensor slot_mapping, float k_scale, float v_scale) -> ()");
}

TORCH_LIBRARY_IMPL(winintel_sm70, CUDA, m) {
  m.impl("gathered_qk_norm", &winintel_gathered_qk_norm);
  m.impl("fused_moe", &fused_moe);
  m.impl("paged_attention", &paged_attention);
  m.impl("reshape_and_cache", &reshape_and_cache);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "WinIntel SM70 attention kernels";
}
