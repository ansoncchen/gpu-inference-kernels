#include "q4.h"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace q4 {

float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
  if (exp == 0) {  // zero or subnormal: value = man * 2^-24
    float f = std::ldexp((float)man, -24);
    return sign ? -f : f;
  }
  uint32_t bits = exp == 31 ? (sign | 0x7F800000u | (man << 13))
                            : (sign | ((exp + 112) << 23) | (man << 13));
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

uint16_t float_to_half(float f) {  // round to nearest, ties to even
  uint32_t b;
  std::memcpy(&b, &f, 4);
  uint16_t sign = (b >> 16) & 0x8000;
  float a = std::fabs(f);
  if (!(a <= 3.40282347e38f)) return sign | (std::isnan(f) ? 0x7E00 : 0x7C00);
  if (a < std::ldexp(1.0f, -14)) {  // subnormal half
    return sign | (uint16_t)std::lrint(std::ldexp((double)a, 24));
  }
  int e;
  float m = std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
  long r = std::lrint(std::ldexp((double)m, 11));  // 11 significant bits
  if (r == 2048) { r = 1024; e++; }
  if (e > 16) return sign | 0x7C00;
  return sign | (uint16_t)(((e + 14) << 10) | (r - 1024));
}

static void check_shape(int rows, int cols) {
  if (rows <= 0 || cols <= 0 || cols % kGroup) throw std::invalid_argument("cols must be a positive multiple of 32");
}

Matrix quantize(const float* a, int rows, int cols, bool search) {
  check_shape(rows, cols);
  Matrix m;
  m.rows = rows, m.cols = cols;
  m.w.assign((size_t)rows * cols / 2, 0);
  m.scale.assign((size_t)rows * cols / kGroup, 0);
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; r++) {
    for (int g = 0; g < cols / kGroup; g++) {
      const float* p = a + (size_t)r * cols + g * kGroup;
      float extreme = 0;
      for (int j = 0; j < kGroup; j++)
        if (std::fabs(p[j]) > std::fabs(extreme)) extreme = p[j];
      // The largest-magnitude weight maps exactly to -8 (same rule as ggml Q4_0).
      uint16_t h = float_to_half(extreme / -8.0f);
      if (search && extreme != 0) {
        double best = 1e300;
        for (int sign = -1; sign <= 1; sign += 2)
          for (int i = 0; i <= 24; i++) {
            uint16_t hc = float_to_half(extreme * sign / (8.0f + 0.25f * (i - 4)));
            float dc = half_to_float(hc);
            if (dc == 0) continue;
            double e = 0;
            for (int j = 0; j < kGroup; j++) {
              float q = std::nearbyint(p[j] / dc);
              q = q < -8 ? -8 : q > 7 ? 7 : q;
              e += (double)(p[j] - q * dc) * (p[j] - q * dc);
            }
            if (e < best) best = e, h = hc;
          }
      }
      float d = half_to_float(h), inv = d != 0 ? 1.0f / d : 0.0f;
      m.scale[(size_t)r * m.groups() + g] = h;
      for (int j = 0; j < kGroup; j++) {
        long q = std::lrint(p[j] * inv);
        q = q < -8 ? -8 : q > 7 ? 7 : q;
        size_t c = (size_t)g * kGroup + j;
        m.w[(size_t)r * (cols / 2) + c / 2] |= uint8_t(q + 8) << ((c & 1) ? 4 : 0);
      }
    }
  }
  return m;
}

Matrix random_matrix(int rows, int cols, uint64_t seed) {
  check_shape(rows, cols);
  Matrix m;
  m.rows = rows, m.cols = cols;
  m.w.resize((size_t)rows * cols / 2);
  m.scale.resize((size_t)rows * cols / kGroup);
  std::mt19937_64 rng(seed);
  for (size_t i = 0; i < m.w.size(); i += 8) {
    uint64_t v = rng();
    std::memcpy(&m.w[i], &v, std::min<size_t>(8, m.w.size() - i));
  }
  std::normal_distribution<float> n(0.0f, 0.01f);
  for (auto& s : m.scale) s = float_to_half(n(rng));  // signed, like ggml scales
  return m;
}

std::vector<float> dequantize(const Matrix& m) {
  std::vector<float> a((size_t)m.rows * m.cols);
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++)
    for (int c = 0; c < m.cols; c++) a[(size_t)r * m.cols + c] = m.s(r, c) * m.q(r, c);
  return a;
}

Tiled to_tiled(const Matrix& m) {
  Tiled t;
  t.rows = m.rows, t.cols = m.cols;
  t.tile_rows = (m.rows + kTileRows - 1) / kTileRows;
  t.tile_cols = (m.cols + kTileCols - 1) / kTileCols;
  size_t tiles = (size_t)t.tile_rows * t.tile_cols;
  t.w.assign(tiles * kTileRows * kTileGroups * 16, 0x88);  // padding weights are q = 0
  t.scale.assign(tiles * kTileRows * kTileGroups, 0);      // padding scales are 0
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++) {
    int tr = r / kTileRows, lr = r % kTileRows;
    for (int g = 0; g < m.groups(); g++) {
      size_t tile = (size_t)tr * t.tile_cols + g / kTileGroups;
      int lg = g % kTileGroups;
      std::memcpy(&t.w[(tile * kTileRows * kTileGroups + (size_t)lr * kTileGroups + lg) * 16],
                  &m.w[(size_t)r * (m.cols / 2) + (size_t)g * 16], 16);
      t.scale[tile * kTileRows * kTileGroups + ((size_t)(lr / 8) * kTileGroups + lg) * 8 + lr % 8] =
          m.scale[(size_t)r * m.groups() + g];
    }
  }
  return t;
}

Octet to_octet(const Matrix& m) {
  Octet t;
  t.rows = m.rows, t.cols = m.cols;
  t.octets = (m.rows + 7) / 8;
  t.chunks = (m.cols + 127) / 128;
  t.w.assign((size_t)t.octets * t.chunks * 512, 0x88);
  t.scale.assign((size_t)t.octets * t.chunks * 4 * 8, 0);
#pragma omp parallel for schedule(static)
  for (int o = 0; o < t.octets; o++)
    for (int f = 0; f < 8; f++) {
      int r = o * 8 + f;
      if (r >= m.rows) break;
      const uint8_t* src = &m.w[(size_t)r * (m.cols / 2)];
      for (int c4 = 0; c4 < m.cols / 4; c4++)  // two bytes = 4 columns
        std::memcpy(&t.w[(((size_t)o * t.chunks * 32) + c4) * 16 + f * 2], src + c4 * 2, 2);
      for (int g = 0; g < m.groups(); g++)
        t.scale[((size_t)o * t.chunks * 4 + g) * 8 + f] = m.scale[(size_t)r * m.groups() + g];
    }
  return t;
}

void gemv_reference(const Matrix& m, const float* x, double* y, double* bound) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++) {
    double acc = 0, b = 0;
    for (int g = 0; g < m.groups(); g++) {
      double s = m.s(r, g * kGroup), gs = 0, gb = 0;
      for (int j = 0; j < kGroup; j++) {
        double xj = x[g * kGroup + j];
        gs += m.q(r, g * kGroup + j) * xj;
        gb += 8 * std::fabs(xj);
      }
      acc += s * gs, b += std::fabs(s) * gb;
    }
    y[r] = acc;
    if (bound) bound[r] = b;
  }
}

}  // namespace q4
