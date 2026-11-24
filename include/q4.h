// INT4 group-wise symmetric weight format, CPU reference and packing helpers.
//
// A matrix has `rows` x `cols` weights, row-major, with cols a multiple of 32.
// Each run of 32 consecutive weights in a row (a "group") shares one FP16 scale:
//     w = scale * q,   q in [-8, 7],   stored as the unsigned nibble u = q + 8.
// That is 18 bytes per 32 weights = 0.5625 bytes/weight (FP16 is 2, FP32 is 4).
// The numbers are the same as ggml's Q4_0; only the memory layout differs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace q4 {

constexpr int kGroup = 32;            // weights per scale
constexpr int kTileRows = 32;         // tiled layout: rows per tile
constexpr int kTileGroups = 32;       // tiled layout: groups (warp lanes) per tile
constexpr int kTileCols = kGroup * kTileGroups;  // 1024 columns

uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

struct Matrix {
  int rows = 0, cols = 0;
  // Row-major. Nibble k of a row lives in byte k/2 (low nibble first).
  std::vector<uint8_t> w;        // rows * cols / 2 bytes
  std::vector<uint16_t> scale;   // rows * cols / 32 FP16 scales

  int groups() const { return cols / kGroup; }
  size_t bytes() const { return w.size() + scale.size() * sizeof(uint16_t); }
  int q(int r, int c) const {
    uint8_t b = w[(size_t)r * (cols / 2) + c / 2];
    return int((c & 1) ? (b >> 4) : (b & 0xF)) - 8;
  }
  float s(int r, int c) const { return half_to_float(scale[(size_t)r * groups() + c / kGroup]); }
};

// Tiled layout used by the fast GPU kernels. The matrix is padded to whole
// 32-row x 1024-column tiles. Inside a tile:
//   weights: [row 0..31][group 0..31] 16-byte blocks (16 KB)
//   scales : [k 0..3][group 0..31] 16-byte blocks, each holding the FP16 scales
//            of 8 consecutive rows (8k..8k+7) for that group (2 KB)
// so lane g of a warp always reads 16 aligned bytes, and the 32 lanes together
// always read 512 contiguous bytes.
struct Tiled {
  int rows = 0, cols = 0, tile_rows = 0, tile_cols = 0;
  std::vector<uint8_t> w;
  std::vector<uint16_t> scale;
  size_t bytes() const { return w.size() + scale.size() * sizeof(uint16_t); }
};

// Octet layout: 8 rows x 4 columns per 16-byte block, so that the 32 lanes of a
// warp read 512 contiguous weight bytes (8 rows x 128 columns) and 512
// contiguous x bytes (128 floats) at the same time.
//   weights: [octet][chunk of 128 cols][lane 0..31] 16-byte blocks; 16-bit field
//            f of a block holds row f, columns 4*lane .. 4*lane+3 (low first)
//   scales : [octet][group][8 rows] FP16
// Rows are padded to a multiple of 8 and columns to a multiple of 128.
struct Octet {
  int rows = 0, cols = 0, octets = 0, chunks = 0;
  std::vector<uint8_t> w;
  std::vector<uint16_t> scale;
  size_t bytes() const { return w.size() + scale.size() * sizeof(uint16_t); }
};
Octet to_octet(const Matrix& m);

// Quantize FP32 row-major weights.
// search = true tries a range of scales per group and keeps the one with the
// smallest squared error instead of always mapping the extreme weight to -8.
Matrix quantize(const float* a, int rows, int cols, bool search = false);
// Random matrix built directly in the quantized domain (all nibbles uniform).
Matrix random_matrix(int rows, int cols, uint64_t seed);
std::vector<float> dequantize(const Matrix& m);
Tiled to_tiled(const Matrix& m);

// Reference y = W x in double precision. If `bound` is given it receives, per
// row, sum_g |scale_g| * 8 * sum_{j in g} |x_j|: the largest possible magnitude
// of the row's terms. Kernel errors are reported relative to this number.
void gemv_reference(const Matrix& m, const float* x, double* y, double* bound = nullptr);

}  // namespace q4
