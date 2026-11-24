// CPU INT4 GEMV: scalar baseline, AVX2+FMA with FP32 x, AVX2 with int8 x.
// Rows are split between OpenMP threads; every thread streams its own rows.
#include "cpu.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace q4 {
namespace {

void run_scalar(const Matrix& m, const float* x, float* y) {
  const int G = m.groups();
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++) {
    const uint8_t* w = &m.w[(size_t)r * (m.cols / 2)];
    const uint16_t* s = &m.scale[(size_t)r * G];
    float acc = 0;
    for (int g = 0; g < G; g++) {
      float d = 0;
      for (int j = 0; j < 16; j++) {
        uint8_t b = w[g * 16 + j];
        d += (float)((b & 0xF) - 8) * x[g * 32 + 2 * j];
        d += (float)((b >> 4) - 8) * x[g * 32 + 2 * j + 1];
      }
      acc += _cvtsh_ss(s[g]) * d;
    }
    y[r] = acc;
  }
}

inline float hsum(__m256 v) {
  __m128 t = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  t = _mm_add_ps(t, _mm_movehl_ps(t, t));
  return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

// Dot products use the unsigned nibble u = q + 8 and subtract 8 * sum(x) per
// group afterwards, the same correction the GPU kernel uses.
void run_avx2(const Matrix& m, const float* x, float* y) {
  const int G = m.groups();
  std::vector<float> c8((G + 7) / 8 * 8, 0.0f);
  for (int g = 0; g < G; g++) {
    float t = 0;
    for (int j = 0; j < 32; j++) t += x[g * 32 + j];
    c8[g] = -8.0f * t;
  }
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++) {
    const uint8_t* w = &m.w[(size_t)r * (m.cols / 2)];
    const uint16_t* s = &m.scale[(size_t)r * G];
    const __m128i mask = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    float corr = 0;
    for (int g = 0; g < G; g++, w += 16) {
      const float* xg = x + g * 32;
      __m128i b = _mm_loadu_si128((const __m128i*)w);
      __m128i lo = _mm_and_si128(b, mask), hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask);
      __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
      auto f = [](__m128i v) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)); };
      __m256 d = _mm256_mul_ps(f(n0), _mm256_loadu_ps(xg));
      d = _mm256_fmadd_ps(f(_mm_srli_si128(n0, 8)), _mm256_loadu_ps(xg + 8), d);
      d = _mm256_fmadd_ps(f(n1), _mm256_loadu_ps(xg + 16), d);
      d = _mm256_fmadd_ps(f(_mm_srli_si128(n1, 8)), _mm256_loadu_ps(xg + 24), d);
      float sc = _cvtsh_ss(s[g]);
      acc = _mm256_fmadd_ps(d, _mm256_set1_ps(sc), acc);
      corr += sc * c8[g];
    }
    y[r] = hsum(acc) + corr;
  }
}

// x is quantized to int8 per group, so one maddubs covers all 32 weights.
// xq is stored as 16 even-column values then 16 odd-column values, which is the
// order the nibbles come out in without any unpacking.
void run_avx2_int8(const Matrix& m, const float* x, float* y) {
  const int G = m.groups(), P = (G + 7) / 8 * 8;
  std::vector<int8_t> xq((size_t)G * 32);
  std::vector<float> sx(P, 0.0f), corr(P, 0.0f);
  for (int g = 0; g < G; g++) {
    float mx = 0;
    for (int j = 0; j < 32; j++) mx = std::max(mx, std::fabs(x[g * 32 + j]));
    float d = mx / 127.0f, inv = d > 0 ? 1.0f / d : 0.0f;
    int sum = 0;
    for (int j = 0; j < 32; j++) {
      int q = (int)std::lrint(x[g * 32 + j] * inv);
      xq[g * 32 + (j & 1) * 16 + j / 2] = (int8_t)q;
      sum += q;
    }
    sx[g] = d;
    corr[g] = -8.0f * d * sum;
  }
#pragma omp parallel for schedule(static)
  for (int r = 0; r < m.rows; r++) {
    const uint8_t* w = &m.w[(size_t)r * (m.cols / 2)];
    const uint16_t* s = &m.scale[(size_t)r * G];
    const __m256i mask = _mm256_set1_epi8(0xF), ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps(), cacc = _mm256_setzero_ps();
    for (int g0 = 0; g0 < G; g0 += 8) {
      __m256 sv;
      if (g0 + 8 <= G) {
        sv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + g0)));
      } else {
        alignas(16) uint16_t t[8] = {0};
        std::copy(s + g0, s + G, t);
        sv = _mm256_cvtph_ps(_mm_load_si128((const __m128i*)t));
      }
      cacc = _mm256_fmadd_ps(sv, _mm256_loadu_ps(&corr[g0]), cacc);
      alignas(32) float ss[8];
      _mm256_store_ps(ss, _mm256_mul_ps(sv, _mm256_loadu_ps(&sx[g0])));
      for (int g = g0; g < std::min(g0 + 8, G); g++) {
        __m128i b = _mm_loadu_si128((const __m128i*)(w + g * 16));
        __m256i n = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(b, 4), b), mask);
        __m256i p = _mm256_maddubs_epi16(n, _mm256_loadu_si256((const __m256i*)&xq[g * 32]));
        __m256 d = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));
        acc = _mm256_fmadd_ps(d, _mm256_broadcast_ss(&ss[g - g0]), acc);
      }
    }
    y[r] = hsum(acc) + hsum(cacc);
  }
}

}  // namespace

void cpu_gemv_fp32(const float* a, int rows, int cols, const float* x, float* y) {
#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; r++) {
    const float* p = a + (size_t)r * cols;
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (int c = 0; c < cols; c += 32) {  // cols is a multiple of 32
      s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p + c), _mm256_loadu_ps(x + c), s0);
      s1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + c + 8), _mm256_loadu_ps(x + c + 8), s1);
      s2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + c + 16), _mm256_loadu_ps(x + c + 16), s2);
      s3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + c + 24), _mm256_loadu_ps(x + c + 24), s3);
    }
    y[r] = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  }
}

double cpu_read(const float* p, size_t n) {
  double total = 0;
#pragma omp parallel reduction(+ : total)
  {
    __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n / 32; i++) {
      const float* q = p + i * 32;
      a = _mm256_add_ps(a, _mm256_loadu_ps(q));
      b = _mm256_add_ps(b, _mm256_loadu_ps(q + 8));
      c = _mm256_add_ps(c, _mm256_loadu_ps(q + 16));
      d = _mm256_add_ps(d, _mm256_loadu_ps(q + 24));
    }
    total += hsum(_mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
  }
  return total;
}

const std::vector<CpuVariant>& cpu_variants() {
  static const std::vector<CpuVariant> v = {
      {"c0_scalar", false, run_scalar, "plain C++ loop, -O3"},
      {"c1_avx2", false, run_avx2, "AVX2 + FMA, FP32 x"},
      {"c2_avx2_int8x", true, run_avx2_int8, "AVX2, x quantized to int8 per group"},
  };
  return v;
}

}  // namespace q4
