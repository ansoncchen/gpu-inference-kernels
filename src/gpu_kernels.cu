// INT4 GEMV kernels, kept one optimization step at a time.
//
//   s0_naive      one thread per row, byte loads, int->float conversion
//   s1_coalesced  one warp per row: the 32 lanes read adjacent bytes
//   s2_vec16      16-byte loads: one lane handles a whole 32-weight group
//   s3_mantissa   build floats from nibble bits instead of converting integers
//   s4_xregs      each lane keeps its 32 x values in registers, warp walks rows
//   s5_tiled      32x1024 tiled layout, vector scale loads, reduce-scatter
//   s6_block      block-level reduction (no memset/atomics) + tuned prefetch
#include "gpu.h"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace q4 {
namespace {

__device__ __forceinline__ float half2float(uint16_t h) { return __half2float(*(__half*)&h); }

__host__ __device__ __forceinline__ uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- s0: naive
// Neighbouring threads read addresses a whole row apart, so each 32-byte DRAM
// sector that is fetched serves one byte to one thread.
__global__ void k_naive(const uint8_t* w, const uint16_t* s, const float* x, float* y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const uint8_t* row = w + (size_t)r * (cols / 2);
  const uint16_t* srow = s + (size_t)r * (cols / kGroup);
  float acc = 0;
  for (int c = 0; c < cols; c += 2) {
    uint32_t b = row[c / 2];
    float sc = half2float(srow[c / kGroup]);
    acc += sc * (float)((int)(b & 0xF) - 8) * x[c];
    acc += sc * (float)((int)(b >> 4) - 8) * x[c + 1];
  }
  y[r] = acc;
}

// ------------------------------------------------------------ s1: coalesced
// A warp owns a row; lane i reads byte i, i+32, ... so one warp-wide load is 32
// contiguous bytes.
__global__ void k_coalesced(const uint8_t* w, const uint16_t* s, const float* x, float* y, int rows, int cols) {
  int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  __shared__ float part[256];
  const uint8_t* row = w + (size_t)warp * (cols / 2);
  const uint16_t* srow = s + (size_t)warp * (cols / kGroup);
  float acc = 0;
  for (int b = lane; b < cols / 2; b += 32) {
    uint32_t v = row[b];
    int c = 2 * b;
    float sc = half2float(srow[c / kGroup]);
    acc += sc * (float)((int)(v & 0xF) - 8) * x[c];
    acc += sc * (float)((int)(v >> 4) - 8) * x[c + 1];
  }
  part[threadIdx.x] = acc;
  __syncwarp();
  if (lane == 0) {
    float sum = 0;
    for (int i = 0; i < 32; i++) sum += part[threadIdx.x + i];
    y[warp] = sum;
  }
}

// ------------------------------------------------- s2 / s3: 16-byte loads
// A lane takes a whole group: one 16-byte weight load, one scale, 32 x values.
__device__ __forceinline__ float dot32_convert(uint4 v, const float* x) {
  float d = 0;
  uint32_t word[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int k = 0; k < 8; k++) d += (float)((int)((word[i] >> (4 * k)) & 0xF) - 8) * x[i * 8 + k];
  return d;
}

// Put nibble u in the top four mantissa bits of 1.0f: that is exactly 1 + u/16,
// made with one shift and one three-input logic op, no conversion unit.
template <int K>
__device__ __forceinline__ float nib2float(uint32_t word) {
  uint32_t t = K <= 4 ? word << (19 - 4 * K) : word >> (4 * K - 19);
  return __uint_as_float((t & 0x00780000u) | 0x3F800000u);
}
// sum (1 + u/16) x = sum x + (1/16) sum u x, so with X = sum x
//   sum (u - 8) x = 16 * (D - X) - 8 X = 16 * (D - 1.5 X)
__device__ __forceinline__ float dot32_raw(uint4 v, const float* x) {
  float d = 0;
  uint32_t word[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t t = word[i];
    d = fmaf(nib2float<0>(t), x[i * 8 + 0], d);
    d = fmaf(nib2float<1>(t), x[i * 8 + 1], d);
    d = fmaf(nib2float<2>(t), x[i * 8 + 2], d);
    d = fmaf(nib2float<3>(t), x[i * 8 + 3], d);
    d = fmaf(nib2float<4>(t), x[i * 8 + 4], d);
    d = fmaf(nib2float<5>(t), x[i * 8 + 5], d);
    d = fmaf(nib2float<6>(t), x[i * 8 + 6], d);
    d = fmaf(nib2float<7>(t), x[i * 8 + 7], d);
  }
  return d;
}

template <bool Mantissa, bool XContig = false>
__global__ void k_vec16(const uint4* w, const uint16_t* s, const float* x, const float* xsum, float* y, int rows,
                        int cols) {
  int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  __shared__ float part[256];
  int groups = cols / kGroup;
  const uint4* row = w + (size_t)warp * groups;
  const uint16_t* srow = s + (size_t)warp * groups;
  float acc = 0;
  for (int g = lane; g < groups; g += 32) {
    uint4 v = row[g];
    float sc = half2float(srow[g]);
    const float* xg = x + g * kGroup;
    float xv[32];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      float4 f = XContig ? ((const float4*)x)[(g / 32 * 8 + k) * 32 + lane] : ((const float4*)xg)[k];
      xv[4 * k] = f.x, xv[4 * k + 1] = f.y, xv[4 * k + 2] = f.z, xv[4 * k + 3] = f.w;
    }
    if (Mantissa)
      acc += sc * 16.0f * (dot32_raw(v, xv) - 1.5f * xsum[g]);
    else
      acc += sc * dot32_convert(v, xv);
  }
  part[threadIdx.x] = acc;
  __syncwarp();
  if (lane == 0) {
    float sum = 0;
    for (int i = 0; i < 32; i++) sum += part[threadIdx.x + i];
    y[warp] = sum;
  }
}

// Per-group sum of x, used by s3 only (later steps get it from registers).
__global__ void k_xsum(const float* x, float* xsum, int groups) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  float t = 0;
  for (int j = 0; j < kGroup; j++) t += x[g * kGroup + j];
  xsum[g] = t;
}

// Ablation helper: rewrite x as [tile][k 0..7][lane] float4 blocks, so that the
// 32 lanes of a warp read 512 contiguous bytes per load instead of 16 bytes from
// each of 32 cache lines. Also produces the per-group sums.
__global__ void k_xtranspose(const float4* x, float4* xt, float* xsum, int groups) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= groups) return;
  float t = 0;
  for (int k = 0; k < 8; k++) {
    float4 f = x[g * 8 + k];
    xt[(g / 32 * 8 + k) * 32 + g % 32] = f;
    t += f.x + f.y + f.z + f.w;
  }
  xsum[g] = t;
}

__device__ __forceinline__ float load_x32(const float* x, float xv[32]) {
  float t = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    float4 f = ((const float4*)x)[k];
    xv[4 * k] = f.x, xv[4 * k + 1] = f.y, xv[4 * k + 2] = f.z, xv[4 * k + 3] = f.w;
    t += f.x + f.y + f.z + f.w;
  }
  return t;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
  return v;
}

// ----------------------------------------------------- s4: x in registers
// A warp owns 32 rows x 1024 columns. Lane g holds x[32g .. 32g+31] and reuses
// it for all 32 rows; each row ends with a 5-step shuffle reduction. Column
// tiles of the same row are combined with atomics, so y must be zeroed first.
__global__ void k_xregs(const uint4* w, const uint16_t* s, const float* x, float* y, int rows, int cols) {
  int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int tcols = ceil_div(cols, kTileCols), trows = ceil_div(rows, kTileRows);
  if (warp >= tcols * trows) return;
  int tr = warp / tcols, tc = warp % tcols;
  int groups = cols / kGroup, g = tc * 32 + lane;
  bool on = g < groups;
  float xv[32], xs = 0;
  if (on) xs = load_x32(x + g * kGroup, xv);
  int r0 = tr * kTileRows, r1 = min(rows, r0 + kTileRows);
  for (int r = r0; r < r1; r++) {
    float p = 0;
    if (on) {
      size_t at = (size_t)r * groups + g;
      uint4 v = w[at];
      p = half2float(s[at]) * 16.0f * (dot32_raw(v, xv) - 1.5f * xs);
    }
    p = warp_sum(p);
    if (lane == 0) atomicAdd(y + r, p);
  }
}

// Reduce-scatter across a warp: on entry p[i] is this lane's share of row i; on
// exit p[0] is the full sum for row `lane`. 31 shuffles in total, against
// 32 rows x 5 shuffles for one reduction per row.
__device__ __forceinline__ float reduce_scatter32(float p[32], int lane) {
#pragma unroll
  for (int n = 16; n > 0; n >>= 1) {
    bool hi = lane & n;
#pragma unroll
    for (int i = 0; i < n; i++) {
      float keep = hi ? p[n + i] : p[i];
      float send = hi ? p[i] : p[n + i];
      p[i] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, n);
    }
  }
  return p[0];
}

// One 32-row tile: returns, in lane i, the tile's contribution to row i.
// Prefetch = how many rows of weights are requested before the arithmetic on
// them starts, i.e. outstanding 512-byte loads per warp.
template <int Prefetch>
__device__ __forceinline__ float tile_dot(const uint4* tw, const uint4* ts, const float xv[32], float xs, bool on,
                                          int lane) {
  float p[32];
  if (on) {
    float corr = -1.5f * xs;
#pragma unroll
    for (int k = 0; k < 4; k++) {  // 8 rows share one 16-byte scale load
      uint4 sv = ts[k * 32 + lane];
      uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
      for (int r0 = 0; r0 < 8; r0 += Prefetch) {
        uint4 v[Prefetch];
#pragma unroll
        for (int i = 0; i < Prefetch; i++) v[i] = tw[(k * 8 + r0 + i) * 32 + lane];
#pragma unroll
        for (int i = 0; i < Prefetch; i++) {
          int r = r0 + i;
          uint16_t h = (r & 1) ? sw[r / 2] >> 16 : sw[r / 2] & 0xFFFF;
          p[k * 8 + r] = half2float(h) * 16.0f * (dot32_raw(v[i], xv) + corr);
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 32; i++) p[i] = 0;
  }
  return reduce_scatter32(p, lane);
}

// ------------------------------------------------------------- s5: tiled
__global__ void k_tiled(const uint4* tw, const uint4* ts, const float* x, float* y, int tcols, int trows, int cols) {
  int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (warp >= tcols * trows) return;
  int tr = warp / tcols, tc = warp % tcols, g = tc * 32 + lane;
  bool on = g < cols / kGroup;
  float xv[32], xs = 0;
  if (on) xs = load_x32(x + g * kGroup, xv);
  float out = tile_dot<1>(tw + (size_t)warp * 1024, ts + (size_t)warp * 128, xv, xs, on, lane);
  atomicAdd(y + tr * kTileRows + lane, out);
}

// ------------------------------------------------------------- s6: block
// A block owns all column tiles of RowTiles consecutive row tiles (one warp per
// column tile), so the final sum happens in shared memory and y is written once,
// with no memset and no atomics. x stays in registers for RowTiles * 32 rows.
template <int Prefetch, int RowTiles>
__global__ void k_block(const uint4* tw, const uint4* ts, const float* x, float* y, int tcols, int trows, int cols) {
  extern __shared__ float part[];  // [tcols][RowTiles][32]
  int tc = threadIdx.x / 32, lane = threadIdx.x % 32, g = tc * 32 + lane;
  bool on = g < cols / kGroup;
  float xv[32], xs = 0;
  if (on) xs = load_x32(x + g * kGroup, xv);
  int tr0 = blockIdx.x * RowTiles;
  for (int i = 0; i < RowTiles; i++) {
    if (tr0 + i >= trows) break;
    size_t tile = (size_t)(tr0 + i) * tcols + tc;
    part[(tc * RowTiles + i) * 32 + lane] = tile_dot<Prefetch>(tw + tile * 1024, ts + tile * 128, xv, xs, on, lane);
  }
  __syncthreads();
  // Warp j sums row tile j; with fewer warps than row tiles, warps wrap around.
  for (int i = tc; i < RowTiles; i += tcols) {
    if (tr0 + i >= trows) break;
    float sum = 0;
    for (int c = 0; c < tcols; c++) sum += part[(c * RowTiles + i) * 32 + lane];
    y[(tr0 + i) * kTileRows + lane] = sum;
  }
}

// --------------------------------------------------------- s7: short loop
// Same block-level scheme, but the body covers only B rows and is looped, so
// the hot loop stays small enough for the instruction caches. H = rows per
// block (how long x stays in registers).
//
// Reduction for a batch of B rows: log2(B) reduce-scatter steps leave lane l
// with a partial sum for row l % B, then all-reduce steps over the remaining
// lane bits finish it. B = 4: 3 + 3 = 6 shuffles for 4 rows instead of 20.
template <int B>
__device__ __forceinline__ float reduce_batch(float p[B], int lane) {
#pragma unroll
  for (int n = B / 2; n > 0; n >>= 1) {
    bool hi = lane & n;
#pragma unroll
    for (int i = 0; i < n; i++) {
      float keep = hi ? p[n + i] : p[i];
      float send = hi ? p[i] : p[n + i];
      p[i] = keep + __shfl_xor_sync(0xFFFFFFFFu, send, n);
    }
  }
  float v = p[0];
#pragma unroll
  for (int n = B; n < 32; n <<= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, n);
  return v;
}

template <int B>
struct ScaleWord;
template <>
struct ScaleWord<1> {
  using T = uint16_t;
  static __device__ __forceinline__ uint16_t get(T v, int) { return v; }
};
template <>
struct ScaleWord<2> {
  using T = uint32_t;
  static __device__ __forceinline__ uint16_t get(T v, int i) { return i ? v >> 16 : v & 0xFFFF; }
};
template <>
struct ScaleWord<4> {
  using T = uint2;
  static __device__ __forceinline__ uint16_t get(T v, int i) {
    uint32_t w = i < 2 ? v.x : v.y;
    return (i & 1) ? w >> 16 : w & 0xFFFF;
  }
};
template <>
struct ScaleWord<8> {
  using T = uint4;
  static __device__ __forceinline__ uint16_t get(T v, int i) {
    uint32_t w = i < 2 ? v.x : i < 4 ? v.y : i < 6 ? v.z : v.w;
    return (i & 1) ? w >> 16 : w & 0xFFFF;
  }
};

// 16 * dot - 24 * xsum, folded into one scale multiply per row.
template <int B, bool Convert = false>
__device__ __forceinline__ void batch_dot(const uint4* tw, const uint16_t* ts, const float xv[32], float corr,
                                          int lane, int lr, float p[B]) {
  typename ScaleWord<B>::T sv =
      *(const typename ScaleWord<B>::T*)(ts + ((lr / 8) * 32 + lane) * 8 + lr % 8);
  uint4 v[B];
#pragma unroll
  for (int i = 0; i < B; i++) v[i] = tw[(lr + i) * 32 + lane];
#pragma unroll
  for (int i = 0; i < B; i++)
    p[i] = half2float(ScaleWord<B>::get(sv, i)) *
           (Convert ? dot32_convert(v[i], xv) : fmaf(dot32_raw(v[i], xv), 16.0f, corr));
}

template <int B, int H, bool Convert = false>
__global__ void k_loop(const uint4* tw, const uint16_t* ts, const float* x, float* y, int tcols, int rows, int cols) {
  extern __shared__ float part[];  // [tcols][H]
  int tc = threadIdx.x / 32, lane = threadIdx.x % 32, g = tc * 32 + lane;
  bool on = g < cols / kGroup;
  float xv[32], corr = 0;
  if (on) corr = -24.0f * load_x32(x + g * kGroup, xv);
  int r0 = blockIdx.x * H;
  for (int h = 0; h < H; h += B) {
    int r = r0 + h;
    if (r >= rows) break;
    size_t tile = (size_t)(r / kTileRows) * tcols + tc;
    float p[B];
    if (on)
      batch_dot<B, Convert>(tw + tile * 1024, ts + tile * 1024, xv, corr, lane, r % kTileRows, p);
    else
#pragma unroll
      for (int i = 0; i < B; i++) p[i] = 0;
    float v = reduce_batch<B>(p, lane);
    if (lane < B) part[tc * H + h + lane] = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    if (r0 + i >= rows) break;
    float sum = 0;
    for (int c = 0; c < tcols; c++) sum += part[c * H + i];
    y[r0 + i] = sum;
  }
}

// ---------------------------------------------------------------- launchers
constexpr int kBlock = 256;

void run_naive(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  k_naive<<<ceil_div(m.rows, kBlock), kBlock, 0, st>>>(m.w, m.scale, x, y, m.rows, m.cols);
}
void run_coalesced(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  k_coalesced<<<ceil_div(m.rows, kBlock / 32), kBlock, 0, st>>>(m.w, m.scale, x, y, m.rows, m.cols);
}
void run_vec16(const GpuMat& m, const float* x, float* y, float* scratch, cudaStream_t st) {
  k_vec16<false><<<ceil_div(m.rows, kBlock / 32), kBlock, 0, st>>>((const uint4*)m.w, m.scale, x, scratch, y, m.rows,
                                                                   m.cols);
}
void run_mantissa(const GpuMat& m, const float* x, float* y, float* scratch, cudaStream_t st) {
  int groups = m.cols / kGroup;
  k_xsum<<<ceil_div(groups, 64), 64, 0, st>>>(x, scratch, groups);
  k_vec16<true><<<ceil_div(m.rows, kBlock / 32), kBlock, 0, st>>>((const uint4*)m.w, m.scale, x, scratch, y, m.rows,
                                                                  m.cols);
}
void run_xcontig(const GpuMat& m, const float* x, float* y, float* scratch, cudaStream_t st) {
  int groups = m.cols / kGroup;
  float* xt = scratch + groups + (4 - groups % 4);
  k_xtranspose<<<ceil_div(groups, 64), 64, 0, st>>>((const float4*)x, (float4*)xt, scratch, groups);
  k_vec16<true, true><<<ceil_div(m.rows, kBlock / 32), kBlock, 0, st>>>((const uint4*)m.w, m.scale, xt, scratch, y,
                                                                        m.rows, m.cols);
}
void run_xregs(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  CUDA_CHECK(cudaMemsetAsync(y, 0, (size_t)m.rows * 4, st));
  int warps = ceil_div(m.rows, kTileRows) * ceil_div(m.cols, kTileCols);
  k_xregs<<<ceil_div(warps, kBlock / 32), kBlock, 0, st>>>((const uint4*)m.w, m.scale, x, y, m.rows, m.cols);
}
void run_tiled(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  CUDA_CHECK(cudaMemsetAsync(y, 0, (size_t)m.tile_rows * kTileRows * 4, st));
  int warps = m.tile_rows * m.tile_cols;
  k_tiled<<<ceil_div(warps, kBlock / 32), kBlock, 0, st>>>((const uint4*)m.tw, (const uint4*)m.tscale, x, y,
                                                           m.tile_cols, m.tile_rows, m.cols);
}
template <int Prefetch, int RowTiles>
void run_block(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  // One SM has 64K 32-bit registers. This kernel uses about 100 per thread, so a
  // wide matrix (one warp per 1024 columns) may not fit in one block.
  static const int max_threads = [] {
    cudaFuncAttributes a;
    CUDA_CHECK(cudaFuncGetAttributes(&a, k_block<Prefetch, RowTiles>));
    return a.maxThreadsPerBlock;
  }();
  if (m.tile_cols * 32 > max_threads) {
    run_tiled(m, x, y, nullptr, st);
    return;
  }
  k_block<Prefetch, RowTiles><<<ceil_div(m.tile_rows, RowTiles), m.tile_cols * 32,
                                (size_t)m.tile_cols * RowTiles * 32 * 4, st>>>(
      (const uint4*)m.tw, (const uint4*)m.tscale, x, y, m.tile_cols, m.tile_rows, m.cols);
}

template <int B, int H, bool Convert = false>
void run_loop(const GpuMat& m, const float* x, float* y, float*, cudaStream_t st) {
  static const int max_threads = [] {
    cudaFuncAttributes a;
    CUDA_CHECK(cudaFuncGetAttributes(&a, k_loop<B, H, Convert>));
    return a.maxThreadsPerBlock;
  }();
  if (m.tile_cols * 32 > max_threads) {
    run_tiled(m, x, y, nullptr, st);
    return;
  }
  k_loop<B, H, Convert><<<ceil_div(m.rows, H), m.tile_cols * 32, (size_t)m.tile_cols * H * 4, st>>>(
      (const uint4*)m.tw, m.tscale, x, y, m.tile_cols, m.rows, m.cols);
}

// ------------------------------------------------------------------- probes
template <typename T>
struct Fold;
template <>
struct Fold<uint32_t> {
  static __device__ uint32_t f(uint32_t v) { return v; }
};
template <>
struct Fold<uint2> {
  static __device__ uint32_t f(uint2 v) { return v.x ^ v.y; }
};
template <>
struct Fold<uint4> {
  static __device__ uint32_t f(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }
};
template <>
struct Fold<uint8_t> {
  static __device__ uint32_t f(uint8_t v) { return v; }
};

template <typename T>
__global__ void k_read(const T* p, size_t n, uint32_t* sink) {
  // Each warp streams a contiguous chunk of the buffer; chunks are cyclic so
  // that the running warps always touch different regions.
  size_t warps = (size_t)gridDim.x * (blockDim.x / 32);
  size_t warp = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  size_t per = n / 32 / warps;  // warp-wide loads per warp
  const T* q = p + warp * per * 32 + threadIdx.x % 32;
  uint32_t acc = 0;
  size_t i = 0;
  for (; i + 4 <= per; i += 4) {
    T a = q[i * 32], b = q[(i + 1) * 32], c = q[(i + 2) * 32], d = q[(i + 3) * 32];
    acc ^= Fold<T>::f(a) ^ Fold<T>::f(b) ^ Fold<T>::f(c) ^ Fold<T>::f(d);
  }
  if (acc == 0x9E3779B9u) *sink = acc;  // keeps the loads alive; never true for zeroed data
}
__global__ void k_empty() {}

__global__ void k_fma(float* sink, int iters, float m, float a) {
  float v0 = threadIdx.x, v1 = v0 + 1, v2 = v0 + 2, v3 = v0 + 3, v4 = v0 + 4, v5 = v0 + 5, v6 = v0 + 6, v7 = v0 + 7;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      v0 = fmaf(v0, m, a), v1 = fmaf(v1, m, a), v2 = fmaf(v2, m, a), v3 = fmaf(v3, m, a);
      v4 = fmaf(v4, m, a), v5 = fmaf(v5, m, a), v6 = fmaf(v6, m, a), v7 = fmaf(v7, m, a);
    }
  }
  float t = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
  if (t == 12345.678f) *sink = t;
}

}  // namespace

void launch_read(const void* buf, size_t bytes, int vec, uint32_t* sink, cudaStream_t st) {
  // About 8 KB per warp, at most 16384 warps.
  const int threads = 128, blocks = (int)min((size_t)4096, max((size_t)1, bytes / 32768));
  if (vec == 16) k_read<uint4><<<blocks, threads, 0, st>>>((const uint4*)buf, bytes / 16, sink);
  if (vec == 8) k_read<uint2><<<blocks, threads, 0, st>>>((const uint2*)buf, bytes / 8, sink);
  if (vec == 4) k_read<uint32_t><<<blocks, threads, 0, st>>>((const uint32_t*)buf, bytes / 4, sink);
  if (vec == 1) k_read<uint8_t><<<blocks, threads, 0, st>>>((const uint8_t*)buf, bytes, sink);
}
void launch_empty(cudaStream_t st) { k_empty<<<1, 1, 0, st>>>(); }
double launch_fma(cudaStream_t st, float* sink) {
  const int blocks = 4096, threads = 256, iters = 512;
  k_fma<<<blocks, threads, 0, st>>>(sink, iters, 0.999f, 0.001f);
  return (double)blocks * threads * iters * 64;
}

const std::vector<Variant>& variants() {
  static const std::vector<Variant> v = {
      {"s0_naive", false, run_naive, "one thread per row, byte loads, int->float"},
      {"s1_coalesced", false, run_coalesced, "one warp per row, contiguous byte loads"},
      {"s2_vec16", false, run_vec16, "16-byte loads, one group per lane"},
      {"s3_mantissa", false, run_mantissa, "nibble in mantissa + group x-sum correction"},
      {"s4_xregs", false, run_xregs, "x in registers, shuffle reduction per row"},
      {"s5_tiled", true, run_tiled, "tiled layout, vector scales, reduce-scatter"},
      {"s6_block", true, run_block<2, 2>, "block-level reduction, prefetch 2, 64 rows/block"},
      {"s7_loop", true, run_loop<8, 16>, "short hot loop: 8 rows per iteration, 16 rows per block"},
      // ablations, each differs from a step above in one way
      {"a3_xcontig", false, run_xcontig, "s3 with x pre-transposed so warp loads are contiguous"},
      {"a7_convert", true, run_loop<8, 16, true>, "s7 with int->float conversion instead of mantissa"},
      // tuning grid: pN = rows of weights in flight per warp, rM = row tiles per block
      {"t_p1_r1", true, run_block<1, 1>, ""},
      {"t_p1_r2", true, run_block<1, 2>, ""},
      {"t_p1_r4", true, run_block<1, 4>, ""},
      {"t_p2_r1", true, run_block<2, 1>, ""},
      {"t_p2_r2", true, run_block<2, 2>, ""},
      {"t_p2_r4", true, run_block<2, 4>, ""},
      {"t_p4_r1", true, run_block<4, 1>, ""},
      {"t_p4_r2", true, run_block<4, 2>, ""},
      {"t_p4_r4", true, run_block<4, 4>, ""},
      {"t_p8_r1", true, run_block<8, 1>, ""},
      {"t_p8_r2", true, run_block<8, 2>, ""},
      {"t_loop_b1_h8", true, run_loop<1, 8>, ""},
      {"t_loop_b1_h16", true, run_loop<1, 16>, ""},
      {"t_loop_b1_h32", true, run_loop<1, 32>, ""},
      {"t_loop_b1_h64", true, run_loop<1, 64>, ""},
      {"t_loop_b1_h128", true, run_loop<1, 128>, ""},
      {"t_loop_b2_h8", true, run_loop<2, 8>, ""},
      {"t_loop_b2_h16", true, run_loop<2, 16>, ""},
      {"t_loop_b2_h32", true, run_loop<2, 32>, ""},
      {"t_loop_b2_h64", true, run_loop<2, 64>, ""},
      {"t_loop_b2_h128", true, run_loop<2, 128>, ""},
      {"t_loop_b4_h8", true, run_loop<4, 8>, ""},
      {"t_loop_b4_h16", true, run_loop<4, 16>, ""},
      {"t_loop_b4_h32", true, run_loop<4, 32>, ""},
      {"t_loop_b4_h64", true, run_loop<4, 64>, ""},
      {"t_loop_b4_h128", true, run_loop<4, 128>, ""},
      {"t_loop_b8_h8", true, run_loop<8, 8>, ""},
      {"t_loop_b8_h16", true, run_loop<8, 16>, ""},
      {"t_loop_b8_h32", true, run_loop<8, 32>, ""},
      {"t_loop_b8_h64", true, run_loop<8, 64>, ""},
      {"t_loop_b8_h128", true, run_loop<8, 128>, ""},
  };
  return v;
}

}  // namespace q4
