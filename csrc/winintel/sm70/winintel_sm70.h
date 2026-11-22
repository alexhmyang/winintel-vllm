// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
//
// WinIntel SM70 attention primitives.
//
// The routines below are written from scratch against the Volta (SM70)
// instruction set. They depend only on CUDA runtime, FP16 and WMMA headers
// so that the same translation unit can be reused
// by the PyTorch extension and by the standalone kernel test harness.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace winintel {
namespace sm70 {

// Paged decode / chunked-prefill attention.
//
// Tensors are fp16 unless noted otherwise.
//   query        [num_tokens, num_heads, head_size]
//   kv_cache     [num_blocks, block_size, num_kv_heads, 2 * head_size]
//                K occupies [0, head_size), V occupies [head_size,
//                2*head_size).
//   out          [num_tokens, num_heads, head_size]
//   block_tables int32 [num_seqs, max_blocks_per_seq]
//   token_seq    int32 [num_tokens] sequence index of every query token
//   token_pos    int32 [num_tokens] absolute position of every query token
//   partial_*    fp32 workspaces used when num_partitions > 1
//                partial_acc [num_tokens, num_heads, num_partitions, head_size]
//                Rounded attention caches scaled scores here before values.
//                partial_max [num_tokens, num_heads, num_partitions]
//                partial_sum [num_tokens, num_heads, num_partitions]
struct PagedAttentionArgs {
  const void* query;
  const void* kv_cache;
  void* out;
  const int* block_tables;
  const int* token_seq;
  const int* token_pos;
  float* partial_acc;
  float* partial_max;
  float* partial_sum;
  int num_tokens;
  int num_heads;
  int num_kv_heads;
  int head_size;
  int block_size;
  int max_blocks_per_seq;
  int num_partitions;
  float scale;
  float softcap;                   // <= 0 disables soft capping
  int sliding_window;              // <= 0 disables the window
  int64_t cache_block_stride = 0;  // In elements; zero means contiguous.
  // Requires one partition and D=64/128; mixed-sequence tiles use scalar work.
  bool tiled_prefill = false;
  bool hf_rounding = false;
  bool cooperative_decode = false;
  bool fp8_cache = false;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
};

cudaError_t paged_attention(const PagedAttentionArgs& args,
                            cudaStream_t stream);

// Packed KV cache write.
//
//   key/value    fp16 [num_tokens, num_kv_heads, head_size]
//   kv_cache     fp16 packed exactly like PagedAttentionArgs::kv_cache
//   slot_mapping int64 [num_tokens]; negative entries are skipped.
struct CacheUpdateArgs {
  const void* key;
  const void* value;
  void* kv_cache;
  const int64_t* slot_mapping;
  int num_tokens;
  int num_kv_heads;
  int head_size;
  int block_size;
  float k_scale;
  float v_scale;
  int64_t cache_block_stride = 0;
  bool fp8_cache = false;
};

cudaError_t reshape_and_cache(const CacheUpdateArgs& args, cudaStream_t stream);

// Routed FP16 SwiGLU experts. Workspaces are [tokens, topk, intermediate]
// and [tokens, topk, hidden]; weights are [experts, 2*intermediate, hidden]
// and [experts, hidden, intermediate].
struct MoeArgs {
  const void* input;
  const void* w13;
  const void* w2;
  const float* weights;
  const int* ids;
  const int* expert_map;
  void* gated;
  void* projected;
  void* out;
  int tokens;
  int hidden;
  int intermediate;
  int experts;
  int global_experts;
  int topk;
  bool weight_on_input;
  bool hf_rounding = false;
};

cudaError_t fused_moe(const MoeArgs& args, cudaStream_t stream);

}  // namespace sm70
}  // namespace winintel
