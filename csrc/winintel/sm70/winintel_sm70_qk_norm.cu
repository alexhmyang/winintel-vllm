// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 WinIntel contributors.
// Author: 数萌AI服务器 (winintel)

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>

namespace {

__global__ void gathered_qk_norm_kernel(const half* input, const half* qw,
                                        const half* kw, half* qout, half* kout,
                                        int qsize, int ksize, int ranks,
                                        int rank, float epsilon) {
  const bool key = blockIdx.y == 1;
  const int local = key ? ksize : qsize;
  const int hidden = local * ranks;
  const int offset = key ? qsize : 0;
  const half* row = input + int64_t(blockIdx.x) * (qsize + ksize) * ranks;
  float variance = 0.0f;
  for (int vec = threadIdx.x; vec < hidden / 8; vec += blockDim.x) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int index = vec * 8 + j;
      const int mapped =
          (index / local) * (qsize + ksize) + offset + index % local;
      const float value = __half2float(row[mapped]);
      variance += value * value;
    }
  }
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float scale;
  variance = Reduce(storage).Reduce(variance, cub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) scale = rsqrtf(variance / hidden + epsilon);
  __syncthreads();
  half* out = (key ? kout : qout) + int64_t(blockIdx.x) * local;
  const half* weight = (key ? kw : qw) + rank * local;
  const half* source = row + rank * (qsize + ksize) + offset;
  for (int index = threadIdx.x; index < local; index += blockDim.x) {
    out[index] = __hmul(__float2half_rn(__half2float(source[index]) * scale),
                        weight[index]);
  }
}

}  // namespace

void winintel_gathered_qk_norm(torch::Tensor input, torch::Tensor qw,
                               torch::Tensor kw, torch::Tensor qout,
                               torch::Tensor kout, int64_t rank,
                               double epsilon) {
  for (const auto& tensor : {input, qw, kw, qout, kout}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.scalar_type() == torch::kHalf,
                "gathered Q/K normalization requires contiguous CUDA float16");
    TORCH_CHECK(tensor.device() == input.device(), "Q/K device mismatch");
  }
  TORCH_CHECK(input.dim() == 2 && qw.dim() == 1 && kw.dim() == 1 &&
                  qout.dim() == 2 && kout.dim() == 2,
              "invalid Q/K normalization dimensions");
  const int64_t qs = qout.size(1), ks = kout.size(1);
  TORCH_CHECK(qs > 0 && ks > 0 && qs % 8 == 0 && ks % 8 == 0,
              "local Q/K widths must be positive multiples of eight");
  const int64_t ranks = qw.numel() / qs;
  TORCH_CHECK(
      ranks > 0 && rank >= 0 && rank < ranks && qw.numel() == ranks * qs &&
          kw.numel() == ranks * ks && input.size(1) == ranks * (qs + ks) &&
          qout.size(0) == input.size(0) && kout.size(0) == input.size(0),
      "gathered Q/K normalization shape or rank mismatch");
  TORCH_CHECK(qw.numel() <= 65536 && kw.numel() <= 65536 &&
                  input.size(0) <= 65535 && epsilon > 0,
              "unsupported Q/K normalization size or epsilon");
  if (input.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(input.device());
  const int threads =
      std::min<int64_t>(std::max(qw.numel(), kw.numel()) / 8, 1024);
  // Matching widths use the same reduction tree as the original RMSNorm.
  TORCH_CHECK(qw.numel() == kw.numel(),
              "fused Q/K normalization requires matching global widths");
  gathered_qk_norm_kernel<<<dim3(input.size(0), 2), threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(input.data_ptr()),
      reinterpret_cast<const half*>(qw.data_ptr()),
      reinterpret_cast<const half*>(kw.data_ptr()),
      reinterpret_cast<half*>(qout.data_ptr()),
      reinterpret_cast<half*>(kout.data_ptr()), qs, ks, ranks, rank, epsilon);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
