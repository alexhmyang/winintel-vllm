// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)
// Email: service@winintel.com
// Website: https://winintel.com
//
// Standalone correctness harness for the WinIntel SM70 kernels.
//
// The harness is deliberately free of PyTorch and of any GPU newer than
// Volta: it compiles with
//   nvcc -gencode arch=compute_70,code=sm_70 \
//        -gencode arch=compute_70,code=compute_70 ...
// and therefore also runs on Ampere/Ada/Blackwell cards through PTX JIT.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "winintel_sm70.h"

namespace {

struct TestCase {
  int num_seqs;
  int num_heads;
  int num_kv_heads;
  int head_size;
  int block_size;
  int partitions;
  int sliding_window;
  bool chunked_prefill;
};

std::mt19937 rng(20261117u);

float random_value() {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(rng);
}

__half to_half(float value) { return __float2half_rn(value); }

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
  }
}

// CPU reference that mirrors the kernel semantics.
void reference_attention(const std::vector<__half>& query,
                         const std::vector<__half>& k_dense,
                         const std::vector<__half>& v_dense,
                         const std::vector<int>& token_seq,
                         const std::vector<int>& token_pos,
                         const std::vector<int>& seq_lens,
                         const TestCase& tc, int max_len, float scale,
                         float softcap, std::vector<__half>& out) {
  const int hs = tc.head_size;
  const int kvh = tc.num_kv_heads;
  const int group = tc.num_heads / tc.num_kv_heads;
  const int tokens = static_cast<int>(token_seq.size());
  out.assign(static_cast<size_t>(tokens) * tc.num_heads * hs, to_half(0.0f));

  for (int t = 0; t < tokens; ++t) {
    const int seq = token_seq[t];
    const int pos = token_pos[t];
    const int start = tc.sliding_window > 0
                          ? std::max(0, pos - tc.sliding_window + 1)
                          : 0;
    (void)seq_lens;
    for (int h = 0; h < tc.num_heads; ++h) {
      const int kv_head = h / group;
      const int count = pos - start + 1;
      std::vector<double> scores(count);
      for (int i = 0; i < count; ++i) {
        const int p = start + i;
        const size_t k_off =
            ((static_cast<size_t>(seq) * max_len + p) * kvh + kv_head) * hs;
        double dot = 0.0;
        for (int d = 0; d < hs; ++d) {
          dot += static_cast<double>(__half2float(
                     query[(static_cast<size_t>(t) * tc.num_heads + h) * hs +
                           d])) *
                 static_cast<double>(__half2float(k_dense[k_off + d]));
        }
        double s = dot * scale;
        if (softcap > 0.0f) {
          s = softcap * std::tanh(s / softcap);
        }
        scores[i] = s;
      }
      const double max_score =
          *std::max_element(scores.begin(), scores.end());
      double sum = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
      }
      for (int d = 0; d < hs; ++d) {
        double acc = 0.0;
        for (int i = 0; i < count; ++i) {
          const int p = start + i;
          const size_t v_off =
              ((static_cast<size_t>(seq) * max_len + p) * kvh + kv_head) * hs;
          acc += scores[i] * static_cast<double>(__half2float(v_dense[v_off + d]));
        }
        acc = sum > 0.0 ? acc / sum : 0.0;
        out[(static_cast<size_t>(t) * tc.num_heads + h) * hs + d] =
            to_half(static_cast<float>(acc));
      }
    }
  }
}

void run_case(const TestCase& tc) {
  const int hs = tc.head_size;
  const int kvh = tc.num_kv_heads;
  const int bs = tc.block_size;
  const int max_len = 96 + static_cast<int>(rng() % 192);

  std::vector<int> seq_lens(tc.num_seqs);
  std::vector<int> q_lens(tc.num_seqs);
  for (int s = 0; s < tc.num_seqs; ++s) {
    seq_lens[s] = 1 + static_cast<int>(rng() % max_len);
    if (tc.chunked_prefill) {
      q_lens[s] = 1 + static_cast<int>(rng() % std::min(7, seq_lens[s]));
    } else {
      q_lens[s] = 1;
    }
  }

  int tokens = 0;
  for (int length : q_lens) {
    tokens += length;
  }

  const int max_blocks = (max_len + bs - 1) / bs + 1;
  const int total_blocks = tc.num_seqs * max_blocks;

  // Shuffle physical block ids so that the block table is not the identity.
  std::vector<int> physical(total_blocks);
  for (int i = 0; i < total_blocks; ++i) {
    physical[i] = i;
  }
  std::shuffle(physical.begin(), physical.end(), rng);

  std::vector<int> block_tables(static_cast<size_t>(tc.num_seqs) * max_blocks);
  for (int s = 0; s < tc.num_seqs; ++s) {
    for (int b = 0; b < max_blocks; ++b) {
      block_tables[static_cast<size_t>(s) * max_blocks + b] =
          physical[static_cast<size_t>(s) * max_blocks + b];
    }
  }

  std::vector<__half> query(static_cast<size_t>(tokens) * tc.num_heads * hs);
  for (auto& value : query) {
    value = to_half(random_value());
  }

  // Dense K/V used by the reference and as the source of the cache update.
  std::vector<__half> k_dense(static_cast<size_t>(tc.num_seqs) * max_len * kvh *
                              hs);
  std::vector<__half> v_dense(k_dense.size());
  for (auto& value : k_dense) {
    value = to_half(random_value() * 0.25f);
  }
  for (auto& value : v_dense) {
    value = to_half(random_value());
  }

  const int total_blocks_capacity = total_blocks;
  std::vector<__half> cache(static_cast<size_t>(total_blocks_capacity) * bs *
                            kvh * 2 * hs);
  std::vector<__half> cache_expected(cache.size(), to_half(0.0f));
  std::vector<__half> cache_from_kernel(cache.size(), to_half(0.0f));

  std::vector<int> token_seq;
  std::vector<int> token_pos;
  std::vector<int64_t> slot_mapping;

  int history_tokens = 0;
  for (int length : seq_lens) {
    history_tokens += length;
  }
  std::vector<__half> key_in(static_cast<size_t>(history_tokens) * kvh * hs);
  std::vector<__half> value_in(key_in.size());

  int cursor = 0;
  for (int s = 0; s < tc.num_seqs; ++s) {
    for (int pos = 0; pos < seq_lens[s]; ++pos) {
      const int block = block_tables[static_cast<size_t>(s) * max_blocks +
                                     pos / bs];
      slot_mapping.push_back(static_cast<int64_t>(block) * bs + pos % bs);
      for (int h = 0; h < kvh; ++h) {
        for (int d = 0; d < hs; ++d) {
          const size_t src =
              ((static_cast<size_t>(s) * max_len + pos) * kvh + h) * hs + d;
          key_in[(static_cast<size_t>(cursor) * kvh + h) * hs + d] =
              k_dense[src];
          value_in[(static_cast<size_t>(cursor) * kvh + h) * hs + d] =
              v_dense[src];
        }
      }
      ++cursor;
    }
    for (int i = 0; i < q_lens[s]; ++i) {
      token_seq.push_back(s);
      token_pos.push_back(seq_lens[s] - q_lens[s] + i);
    }
  }

  // Expected packed cache content (written only at the mapped slots).
  for (size_t t = 0; t < slot_mapping.size(); ++t) {
    const int64_t slot = slot_mapping[t];
    const int block = static_cast<int>(slot / bs);
    const int offset = static_cast<int>(slot % bs);
    for (int h = 0; h < kvh; ++h) {
      for (int d = 0; d < hs; ++d) {
        const size_t dst =
            ((static_cast<size_t>(block) * bs + offset) * kvh + h) * 2 * hs + d;
        cache_expected[dst] =
            key_in[(static_cast<size_t>(t) * kvh + h) * hs + d];
        cache_expected[dst + hs] =
            value_in[(static_cast<size_t>(t) * kvh + h) * hs + d];
      }
    }
  }

  __half* d_query = nullptr;
  __half* d_cache = nullptr;
  __half* d_out = nullptr;
  __half* d_key = nullptr;
  __half* d_value = nullptr;
  int* d_block_tables = nullptr;
  int* d_token_seq = nullptr;
  int* d_token_pos = nullptr;
  int64_t* d_slots = nullptr;
  float* d_partial_acc = nullptr;
  float* d_partial_max = nullptr;
  float* d_partial_sum = nullptr;

  check_cuda(cudaMalloc(&d_query, query.size() * sizeof(__half)), "malloc q");
  check_cuda(cudaMalloc(&d_cache, cache.size() * sizeof(__half)), "malloc kv");
  check_cuda(cudaMalloc(&d_out, query.size() * sizeof(__half)), "malloc out");
  check_cuda(cudaMalloc(&d_key, key_in.size() * sizeof(__half)), "malloc key");
  check_cuda(cudaMalloc(&d_value, value_in.size() * sizeof(__half)),
             "malloc value");
  check_cuda(cudaMalloc(&d_block_tables, block_tables.size() * sizeof(int)),
             "malloc block tables");
  check_cuda(cudaMalloc(&d_token_seq, token_seq.size() * sizeof(int)),
             "malloc token seq");
  check_cuda(cudaMalloc(&d_token_pos, token_pos.size() * sizeof(int)),
             "malloc token pos");
  check_cuda(cudaMalloc(&d_slots, slot_mapping.size() * sizeof(int64_t)),
             "malloc slots");
  if (tc.partitions > 1) {
    const size_t workspace =
        static_cast<size_t>(tokens) * tc.num_heads * tc.partitions * hs;
    check_cuda(cudaMalloc(&d_partial_acc, workspace * sizeof(float)),
               "malloc partial acc");
    check_cuda(
        cudaMalloc(&d_partial_max,
                   static_cast<size_t>(tokens) * tc.num_heads * tc.partitions *
                       sizeof(float)),
        "malloc partial max");
    check_cuda(
        cudaMalloc(&d_partial_sum,
                   static_cast<size_t>(tokens) * tc.num_heads * tc.partitions *
                       sizeof(float)),
        "malloc partial sum");
  }

  check_cuda(cudaMemcpy(d_query, query.data(), query.size() * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "copy q");
  check_cuda(cudaMemcpy(d_cache, cache_from_kernel.data(),
                        cache_from_kernel.size() * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "copy cache");
  check_cuda(cudaMemcpy(d_key, key_in.data(), key_in.size() * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "copy key");
  check_cuda(cudaMemcpy(d_value, value_in.data(),
                        value_in.size() * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "copy value");
  check_cuda(cudaMemcpy(d_block_tables, block_tables.data(),
                        block_tables.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "copy block tables");
  check_cuda(cudaMemcpy(d_token_seq, token_seq.data(),
                        token_seq.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "copy token seq");
  check_cuda(cudaMemcpy(d_token_pos, token_pos.data(),
                        token_pos.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "copy token pos");
  check_cuda(cudaMemcpy(d_slots, slot_mapping.data(),
                        slot_mapping.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "copy slots");

  winintel::sm70::CacheUpdateArgs update{};
  update.key = d_key;
  update.value = d_value;
  update.kv_cache = d_cache;
  update.slot_mapping = d_slots;
  update.num_tokens = history_tokens;
  update.num_kv_heads = kvh;
  update.head_size = hs;
  update.block_size = bs;
  update.k_scale = 1.0f;
  update.v_scale = 1.0f;
  check_cuda(winintel::sm70::reshape_and_cache(update, nullptr),
             "reshape_and_cache");

  check_cuda(cudaMemcpy(cache_from_kernel.data(), d_cache,
                        cache_from_kernel.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "copy cache back");

  double cache_error = 0.0;
  for (size_t i = 0; i < cache.size(); ++i) {
    cache_error = std::max(
        cache_error, std::fabs(static_cast<double>(__half2float(
                                 cache_from_kernel[i])) -
                               static_cast<double>(__half2float(
                                   cache_expected[i]))));
  }
  if (cache_error > 1e-3) {
    std::fprintf(stderr, "cache mismatch for head_size=%d: %g\n", hs,
                 cache_error);
    std::exit(1);
  }

  winintel::sm70::PagedAttentionArgs args{};
  args.query = d_query;
  args.kv_cache = d_cache;
  args.out = d_out;
  args.block_tables = d_block_tables;
  args.token_seq = d_token_seq;
  args.token_pos = d_token_pos;
  args.partial_acc = d_partial_acc;
  args.partial_max = d_partial_max;
  args.partial_sum = d_partial_sum;
  args.num_tokens = tokens;
  args.num_heads = tc.num_heads;
  args.num_kv_heads = kvh;
  args.head_size = hs;
  args.block_size = bs;
  args.max_blocks_per_seq = max_blocks;
  args.num_partitions = tc.partitions;
  args.scale = 1.0f / std::sqrt(static_cast<float>(hs));
  args.softcap = 0.0f;
  args.sliding_window = tc.sliding_window;

  check_cuda(winintel::sm70::paged_attention(args, nullptr),
             "paged_attention");
  check_cuda(cudaDeviceSynchronize(), "sync");

  std::vector<__half> out(query.size());
  check_cuda(cudaMemcpy(out.data(), d_out, out.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "copy out");

  std::vector<__half> expected;
  reference_attention(query, k_dense, v_dense, token_seq, token_pos, seq_lens,
                      tc, max_len, args.scale, args.softcap, expected);

  double max_error = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    max_error = std::max(
        max_error, std::fabs(static_cast<double>(__half2float(out[i])) -
                             static_cast<double>(__half2float(expected[i]))));
  }

  std::printf(
      "seqs=%d heads=%d kv_heads=%d hs=%d bs=%d parts=%d window=%d "
      "chunked=%d tokens=%d -> max_err=%.6f %s\n",
      tc.num_seqs, tc.num_heads, tc.num_kv_heads, hs, bs, tc.partitions,
      tc.sliding_window, static_cast<int>(tc.chunked_prefill), tokens,
      max_error, max_error < 2e-2 ? "PASS" : "FAIL");

  if (max_error >= 2e-2) {
    std::exit(1);
  }

  cudaFree(d_query);
  cudaFree(d_cache);
  cudaFree(d_out);
  cudaFree(d_key);
  cudaFree(d_value);
  cudaFree(d_block_tables);
  cudaFree(d_token_seq);
  cudaFree(d_token_pos);
  cudaFree(d_slots);
  if (d_partial_acc != nullptr) {
    cudaFree(d_partial_acc);
    cudaFree(d_partial_max);
    cudaFree(d_partial_sum);
  }
}

}  // namespace

int main() {
  std::printf("WinIntel SM70 kernel self-test\n");
  run_case({4, 8, 2, 64, 16, 1, 0, false});
  run_case({4, 8, 2, 128, 16, 1, 0, false});
  run_case({4, 8, 2, 128, 16, 8, 0, false});
  run_case({4, 8, 8, 128, 32, 3, 0, false});
  run_case({3, 8, 2, 256, 16, 2, 0, false});
  run_case({4, 8, 2, 128, 16, 1, 64, false});
  run_case({4, 8, 2, 128, 16, 2, 0, true});
  run_case({3, 8, 2, 64, 16, 4, 32, true});
  std::printf("All cases passed.\n");
  return 0;
}
