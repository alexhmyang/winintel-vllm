// SPDX-License-Identifier: Apache-2.0
// Author: 数萌AI服务器 (winintel)
// Reuses the warp reduction and launch pattern of winintel_sm70_w4.cu.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>

__device__ float fp4(unsigned code) {
  unsigned magnitude = code & 7;
  float value = magnitude < 2
      ? magnitude * .5f
      : __uint_as_float(((magnitude >> 1) + 126) << 23) *
            (1.f + (magnitude & 1) * .5f);
  return (code & 8) ? -value : value;
}

__global__ void unpack(const uint8_t* packed, const float* scales, half* out,
                       int columns, int64_t count) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  unsigned bits = packed[i];
  float scale = scales[i / 8];
  out[i * 2] = __float2half_rn(fp4(bits & 15) * scale);
  out[i * 2 + 1] = __float2half_rn(fp4(bits >> 4) * scale);
}

template <int N>
__global__ void linear(const half* x, const uint8_t* packed,
                       const float* scales, half* out, int rows, int columns,
                       int tokens) {
  int lane = threadIdx.x & 31;
  int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (row >= rows) return;
  float sums[N] = {};
  for (int column = lane; column < columns / 2; column += 32) {
    unsigned bits = packed[int64_t(row) * (columns / 2) + column];
    float scale = scales[int64_t(row) * (columns / 16) + column / 8];
    half2 w = __halves2half2(
        __float2half_rn(fp4(bits & 15) * scale),
        __float2half_rn(fp4(bits >> 4) * scale));
    float2 weights = __half22float2(w);
#pragma unroll
    for (int token = 0; token < N; ++token) {
      if (token < tokens) {
        float2 activation = __half22float2(*reinterpret_cast<const half2*>(
            x + int64_t(token) * columns + column * 2));
        sums[token] = fmaf(activation.x, weights.x, sums[token]);
        sums[token] = fmaf(activation.y, weights.y, sums[token]);
      }
    }
  }
#pragma unroll
  for (int token = 0; token < N; ++token) {
    for (int d = 16; d; d >>= 1)
      sums[token] += __shfl_down_sync(0xffffffff, sums[token], d);
    if (!lane && token < tokens)
      out[int64_t(token) * rows + row] = __float2half_rn(sums[token]);
  }
}

// Standard CUDA WMMA tiles; only the active K tile is expanded to FP16.
template <bool Lookup = false>
__global__ void tiled_linear(const half* x, const uint8_t* packed,
                             const float* scales, float* partial, int rows,
                             int columns, int tokens, int partition_tiles,
                             const uint16_t* scale_ids = nullptr,
                             const half* table = nullptr) {
  constexpr int K = 128, STRIDE = 136;
  __shared__ __align__(32) half activation[16 * STRIDE];
  __shared__ __align__(32) half weight[64 * STRIDE];
  __shared__ __align__(32) float result[4 * 256];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> sum;
  nvcuda::wmma::fill_fragment(sum, 0.f);
  for (int tile = 0; tile < partition_tiles; ++tile) {
    int begin = (blockIdx.y * partition_tiles + tile) * K;
    if (begin >= columns) break;
    for (int i = threadIdx.x; i < 16 * K; i += blockDim.x) {
      int token = i / K, column = i % K;
      activation[token * STRIDE + column] =
          token < tokens && begin + column < columns
              ? x[int64_t(token) * columns + begin + column]
              : __float2half_rn(0.f);
    }
    for (int i = threadIdx.x; i < 64 * (K / 2); i += blockDim.x) {
      int local_row = i / (K / 2), column = i % (K / 2);
      int row = blockIdx.x * 64 + local_row;
      half2 pair = __float2half2_rn(0.f);
      if (row < rows && begin + column * 2 < columns) {
        unsigned bits = packed[int64_t(row) * (columns / 2) + begin / 2 + column];
        int64_t group = int64_t(row) * (columns / 16) +
                        begin / 16 + column / 8;
        if constexpr (Lookup) {
          const half* values = table + int(scale_ids[group]) * 16;
          pair = __halves2half2(values[bits & 15], values[bits >> 4]);
        } else {
          float scale = scales[group];
          pair = __halves2half2(
              __float2half_rn(fp4(bits & 15) * scale),
              __float2half_rn(fp4(bits >> 4) * scale));
        }
      }
      *reinterpret_cast<half2*>(weight + local_row * STRIDE + column * 2) = pair;
    }
    __syncthreads();
#pragma unroll
    for (int offset = 0; offset < K; offset += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                             nvcuda::wmma::row_major> a;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                             nvcuda::wmma::col_major> b;
      nvcuda::wmma::load_matrix_sync(a, activation + offset, STRIDE);
      nvcuda::wmma::load_matrix_sync(b, weight + warp * 16 * STRIDE + offset,
                                   STRIDE);
      nvcuda::wmma::mma_sync(sum, a, b, sum);
    }
    __syncthreads();
  }
  nvcuda::wmma::store_matrix_sync(result + warp * 256, sum, 16,
                                 nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < tokens * 16; i += 32) {
    int token = i / 16, row = blockIdx.x * 64 + warp * 16 + i % 16;
    if (row < rows)
      partial[(int64_t(blockIdx.y) * tokens + token) * rows + row] =
          result[warp * 256 + i];
  }
}

__global__ void merge_partitions(const float* partial, half* out, int count,
                                 int partitions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float value = 0;
  for (int part = 0; part < partitions; ++part)
    value += partial[int64_t(part) * count + i];
  out[i] = __float2half_rn(value);
}

void validate(const torch::Tensor& packed, const torch::Tensor& scales) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
                  packed.scalar_type() == torch::kUInt8 && packed.dim() == 2,
              "NVFP4 requires a contiguous CUDA uint8 matrix");
  TORCH_CHECK(packed.size(1) > 0 && packed.size(1) % 8 == 0,
              "NVFP4 columns must be divisible by 16");
  TORCH_CHECK(scales.device() == packed.device() && scales.is_contiguous() &&
                  scales.scalar_type() == torch::kFloat && scales.dim() == 2 &&
                  scales.size(0) == packed.size(0) &&
                  scales.size(1) == packed.size(1) / 8,
              "NVFP4 requires one CUDA float32 scale per 16 weights");
}

torch::Tensor dequant(torch::Tensor packed, torch::Tensor scales) {
  validate(packed, scales);
  const c10::cuda::CUDAGuard guard(packed.device());
  auto out = torch::empty({packed.size(0), packed.size(1) * 2},
                          packed.options().dtype(torch::kHalf));
  if (packed.numel()) {
    unpack<<<(packed.numel() + 255) / 256, 256, 0,
             at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        packed.size(1) * 2, packed.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor scales) {
  validate(packed, scales);
  TORCH_CHECK(x.device() == packed.device() && x.is_contiguous() &&
                  x.scalar_type() == torch::kHalf && x.dim() == 2 &&
                  x.size(1) == packed.size(1) * 2 && x.size(0) <= 8,
              "NVFP4 activations must be a compatible CUDA FP16 matrix");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty({x.size(0), packed.size(0)}, x.options());
  if (x.size(0) && packed.size(0)) {
    dim3 grid((packed.size(0) + 3) / 4);
#define RUN(N) \
    linear<N><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>( \
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), \
        packed.data_ptr<uint8_t>(), scales.data_ptr<float>(), \
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), \
        packed.size(0), x.size(1), x.size(0))
    if (x.size(0) == 1) { RUN(1); }
    else if (x.size(0) <= 3) { RUN(3); }
    else { RUN(8); }
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor gemm(torch::Tensor x, torch::Tensor packed, torch::Tensor scales) {
  validate(packed, scales);
  TORCH_CHECK(x.device() == packed.device() && x.is_contiguous() &&
                  x.scalar_type() == torch::kHalf && x.dim() == 2 &&
                  x.size(1) == packed.size(1) * 2 && x.size(0) <= 16,
              "NVFP4 WMMA requires at most sixteen CUDA FP16 activation rows");
  const c10::cuda::CUDAGuard guard(x.device());
  int rows = packed.size(0), columns = x.size(1), tokens = x.size(0);
  auto out = torch::empty({tokens, rows}, x.options());
  if (tokens && rows) {
    int partitions = rows < 2048 ? 8 : 4;
    int tiles = (columns + 127) / 128;
    int partition_tiles = (tiles + partitions - 1) / partitions;
    auto partial = torch::empty({partitions, tokens, rows},
                                x.options().dtype(torch::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    tiled_linear<false><<<dim3((rows + 63) / 64, partitions), 128, 0, stream>>>(
        reinterpret_cast<half*>(x.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(), scales.data_ptr<float>(),
        partial.data_ptr<float>(), rows, columns, tokens, partition_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_partitions<<<(tokens * rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        tokens * rows, partitions);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

__global__ void unpack_lookup(const uint8_t* packed, const uint16_t* ids,
                              const half* table, half* out, int64_t count) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  unsigned bits = packed[i];
  const half* values = table + int(ids[i / 8]) * 16;
  out[i * 2] = values[bits & 15];
  out[i * 2 + 1] = values[bits >> 4];
}

void validate_lookup(const torch::Tensor& packed, const torch::Tensor& ids,
                     const torch::Tensor& table) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
                  packed.scalar_type() == torch::kUInt8 && packed.dim() == 2 &&
                  packed.size(1) > 0 && packed.size(1) % 8 == 0,
              "NVFP4 requires a contiguous CUDA uint8 matrix");
  TORCH_CHECK(ids.device() == packed.device() && ids.is_contiguous() &&
                  ids.scalar_type() == torch::kShort && ids.dim() == 2 &&
                  ids.size(0) == packed.size(0) &&
                  ids.size(1) == packed.size(1) / 8,
              "NVFP4 lookup requires one int16 ID per group");
  TORCH_CHECK(table.device() == packed.device() && table.is_contiguous() &&
                  table.scalar_type() == torch::kHalf && table.dim() == 3 &&
                  table.size(0) > 0 && table.size(0) <= 128 &&
                  table.size(1) == 256 && table.size(2) == 16,
              "NVFP4 lookup tables must have shape (widths, 256, 16)");
}

torch::Tensor dequant_lookup(torch::Tensor packed, torch::Tensor ids,
                             torch::Tensor table) {
  validate_lookup(packed, ids, table);
  const c10::cuda::CUDAGuard guard(packed.device());
  auto out = torch::empty({packed.size(0), packed.size(1) * 2},
                          packed.options().dtype(torch::kHalf));
  if (packed.numel()) {
    unpack_lookup<<<(packed.numel() + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<uint16_t*>(ids.data_ptr<int16_t>()),
        reinterpret_cast<half*>(table.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), packed.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

torch::Tensor gemm_lookup(torch::Tensor x, torch::Tensor packed,
                          torch::Tensor ids, torch::Tensor table) {
  validate_lookup(packed, ids, table);
  TORCH_CHECK(x.device() == packed.device() && x.is_contiguous() &&
                  x.scalar_type() == torch::kHalf && x.dim() == 2 &&
                  x.size(1) == packed.size(1) * 2 && x.size(0) <= 16,
              "NVFP4 WMMA requires at most sixteen CUDA FP16 activation rows");
  const c10::cuda::CUDAGuard guard(x.device());
  int rows = packed.size(0), columns = x.size(1), tokens = x.size(0);
  auto out = torch::empty({tokens, rows}, x.options());
  if (tokens && rows) {
    int partitions = rows < 2048 ? 8 : 4;
    int tiles = (columns + 127) / 128;
    int partition_tiles = (tiles + partitions - 1) / partitions;
    auto partial = torch::empty({partitions, tokens, rows},
                                x.options().dtype(torch::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    tiled_linear<true><<<dim3((rows + 63) / 64, partitions), 128, 0, stream>>>(
        reinterpret_cast<half*>(x.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(), nullptr, partial.data_ptr<float>(),
        rows, columns, tokens, partition_tiles,
        reinterpret_cast<uint16_t*>(ids.data_ptr<int16_t>()),
        reinterpret_cast<half*>(table.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    merge_partitions<<<(tokens * rows + 255) / 256, 256, 0, stream>>>(
        partial.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        tokens * rows, partitions);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant", &dequant);
  m.def("gemv", &gemv);
  m.def("gemm", &gemm);
  m.def("dequant_lookup", &dequant_lookup);
  m.def("gemm_lookup", &gemm_lookup);
}
