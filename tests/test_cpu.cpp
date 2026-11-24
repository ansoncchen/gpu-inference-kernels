// CPU kernels against the double-precision reference; FP16 and quantizer checks.
#include <cmath>
#include <cstdio>
#include <random>

#include "cpu.h"
#include <immintrin.h>

using namespace q4;
static int failures = 0, checks = 0;
static void expect(bool ok, const char* what) {
  checks++;
  if (!ok) failures++, printf("FAIL %s\n", what);
}

static void run_case(const char* label, const Matrix& m, const std::vector<float>& x) {
  std::vector<double> ref(m.rows), bound(m.rows), qbound(m.rows);
  gemv_reference(m, x.data(), ref.data(), bound.data());
  // int8 x: each x value is off by at most half a step, so a row is off by at
  // most sum_g |scale_g| * 32 * 8 * step_g / 2.
  for (int r = 0; r < m.rows; r++)
    for (int g = 0; g < m.groups(); g++) {
      float mx = 0;
      for (int j = 0; j < 32; j++) mx = std::max(mx, std::fabs(x[g * 32 + j]));
      qbound[r] += std::fabs(m.s(r, g * 32)) * 32 * 8 * (mx / 127 / 2);
    }
  for (const auto& v : cpu_variants()) {
    std::vector<float> y(m.rows, NAN);
    v.run(m, x.data(), y.data());
    double worst = 0;
    bool ok = true;
    for (int r = 0; r < m.rows; r++) {
      double e = std::fabs(y[r] - ref[r]);
      double limit = 1e-6 * bound[r] + (v.int8_x ? qbound[r] : 0);
      if (!(e <= limit)) ok = false;
      if (bound[r] > 0) worst = std::max(worst, e / bound[r]);
    }
    checks++, failures += !ok;
    printf("%-4s %-18s %6dx%-6d %-14s err %.2e\n", ok ? "ok" : "FAIL", label, m.rows, m.cols, v.name.c_str(), worst);
  }
}

int main() {
  // FP16 conversion agrees with the hardware for every half value and a sweep of floats.
  bool same = true, rt = true;
  for (int h = 0; h < 65536; h++) {
    float a = half_to_float((uint16_t)h), b = _cvtsh_ss((uint16_t)h);
    if (std::isnan(b)) continue;
    same &= a == b;
    rt &= float_to_half(a) == (uint16_t)h;
  }
  expect(same, "half_to_float matches F16C");
  expect(rt, "half round trip");
  std::mt19937 rng(5);
  std::normal_distribution<float> n;
  bool conv = true;
  for (int i = 0; i < 1000000; i++) {
    float f = n(rng) * std::pow(10.0f, (int)(rng() % 12) - 8);
    conv &= float_to_half(f) == _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
  }
  expect(conv, "float_to_half matches F16C");

  // Quantizer: the error of each weight is at most half a step (one step for
  // values that clip at +7), and the extreme maps to -8.
  std::vector<float> a(64 * 256);
  for (auto& v : a) v = 0.05f * n(rng);
  Matrix m = quantize(a.data(), 64, 256);
  auto back = dequantize(m);
  bool qok = true;
  for (int r = 0; r < 64; r++)
    for (int c = 0; c < 256; c++) {
      double d = std::fabs(m.s(r, c));
      qok &= std::fabs(back[r * 256 + c] - a[r * 256 + c]) <= (m.q(r, c) == 7 ? 1.01 : 0.501) * d;
    }
  expect(qok, "quantization error within half a step");
  Tiled t = to_tiled(m);
  expect(t.w.size() == 2u * 1 * 16384 && t.scale.size() == 2u * 1 * 1024, "tiled sizes");

  auto rx = [&](int c) {
    std::vector<float> x(c);
    for (auto& v : x) v = n(rng);
    return x;
  };
  int shapes[][2] = {{1, 32}, {33, 1024}, {100, 896}, {257, 1056}, {1000, 2080}, {4096, 4096}, {3, 32768}, {8192, 8192}};
  for (auto& s : shapes) run_case("random", random_matrix(s[0], s[1], 9), rx(s[1]));
  for (int q : {-8, 7}) {
    Matrix e = random_matrix(129, 2048, 3);
    for (auto& b : e.w) b = uint8_t((q + 8) | ((q + 8) << 4));
    run_case(q < 0 ? "all -8" : "all 7", e, rx(2048));
  }
  run_case("quantized", m, rx(256));
  run_case("x = 0", m, std::vector<float>(256, 0.0f));
  auto big = rx(256);
  for (auto& v : big) v = std::fabs(v) * 100;
  run_case("x large positive", m, big);
  printf("%d checks, %d failures\n", checks, failures);
  return failures != 0;
}
