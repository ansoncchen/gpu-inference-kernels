// Compares every GPU variant with the double-precision CPU reference.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "gpu.h"

using namespace q4;

// Max over rows of |y - y_ref| / bound, see gemv_reference. FP32 summation of a
// few thousand terms lands around 1e-9..2e-7 of the bound; 1e-6 is the limit.
static const double kTol = 1e-6;
static int failures = 0, checks = 0;

struct Dev {
  std::vector<void*> ptrs;
  template <class T>
  T* up(const T* src, size_t n, size_t pad = 0) {
    T* p;
    CUDA_CHECK(cudaMalloc(&p, (n + pad) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, (n + pad) * sizeof(T)));
    if (src) CUDA_CHECK(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
    ptrs.push_back(p);
    return p;
  }
  ~Dev() {
    for (void* p : ptrs) cudaFree(p);
  }
};

static void run_case(const char* label, const Matrix& m, const std::vector<float>& x) {
  std::vector<double> ref(m.rows), bound(m.rows);
  gemv_reference(m, x.data(), ref.data(), bound.data());
  Tiled t = to_tiled(m);
  Dev d;
  GpuMat g;
  g.rows = m.rows, g.cols = m.cols, g.tile_rows = t.tile_rows, g.tile_cols = t.tile_cols;
  g.w = d.up(m.w.data(), m.w.size()), g.scale = d.up(m.scale.data(), m.scale.size());
  g.tw = d.up(t.w.data(), t.w.size()), g.tscale = d.up(t.scale.data(), t.scale.size());
  const float* dx = d.up(x.data(), x.size());
  // 32 guard values after y's padded end catch out-of-range writes.
  int ypad = (m.rows + 31) / 32 * 32;
  std::vector<float> y(ypad + 32);
  float* dy = d.up((float*)nullptr, ypad + 32);
  float* scratch = d.up((float*)nullptr, 2 * m.cols + 2048);
  for (const Variant& v : variants()) {
    std::vector<float> fill(ypad + 32, NAN);  // stale output must not leak through
    for (int i = ypad; i < ypad + 32; i++) fill[i] = 777.0f;
    CUDA_CHECK(cudaMemcpy(dy, fill.data(), fill.size() * 4, cudaMemcpyHostToDevice));
    v.launch(g, dx, dy, scratch, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    bool guard = true;
    for (int r = 0; r < m.rows; r++) {
      double e = std::fabs(y[r] - ref[r]);
      e = bound[r] > 0 ? e / bound[r] : e;
      if (!(e <= worst)) worst = e;  // also catches NaN
    }
    for (int i = ypad; i < ypad + 32; i++) guard &= y[i] == 777.0f;
    bool ok = worst <= kTol && guard;
    checks++, failures += !ok;
    printf("%-4s %-26s %6dx%-6d %-12s err %.2e%s\n", ok ? "ok" : "FAIL", label, m.rows, m.cols, v.name.c_str(), worst,
           guard ? "" : " (guard overwritten)");
  }
}

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> n;
  auto rx = [&](int c) {
    std::vector<float> x(c);
    for (auto& v : x) v = n(rng);
    return x;
  };
  // Small, non-tile-aligned, tall, wide, and two larger than the 5.5 MB L2.
  int shapes[][2] = {{1, 32},     {31, 64},     {33, 1024},    {100, 896},   {257, 1056},  {1000, 2080},
                     {4096, 4096}, {11008, 4096}, {3, 32768}, {2048, 33792}, {8192, 8192}, {65, 1536}};
  for (auto& s : shapes) run_case("random", random_matrix(s[0], s[1], s[0] * 31 + s[1]), rx(s[1]));

  {  // every weight at an extreme nibble
    for (int q : {-8, 7}) {
      Matrix m = random_matrix(513, 2048, 3);
      for (auto& b : m.w) b = uint8_t((q + 8) | ((q + 8) << 4));
      run_case(q < 0 ? "all -8" : "all 7", m, rx(2048));
    }
  }
  {  // position test: one non-zero weight per row, x = column index
    Matrix m = random_matrix(2048, 2048, 4);
    std::fill(m.w.begin(), m.w.end(), 0x88);
    std::fill(m.scale.begin(), m.scale.end(), float_to_half(1.0f));
    for (int r = 0; r < m.rows; r++) {
      int c = (r * 37 + 11) % m.cols;
      m.w[(size_t)r * 1024 + c / 2] = (c & 1) ? 0x08 | (0x9 << 4) : 0x80 | 0x9;
    }
    std::vector<float> x(2048);
    for (int i = 0; i < 2048; i++) x[i] = i + 1;
    run_case("position", m, x);
  }
  {  // real quantization path, zero x, constant x
    std::vector<float> a(512 * 1024);
    for (auto& v : a) v = 0.02f * n(rng);
    Matrix m = quantize(a.data(), 512, 1024);
    run_case("quantized", m, rx(1024));
    run_case("x = 0", m, std::vector<float>(1024, 0.0f));
    run_case("x = 1", m, std::vector<float>(1024, 1.0f));
    auto x = rx(1024);
    for (auto& v : x) v = std::fabs(v) * 100;  // no cancellation in x: worst case for s3+
    run_case("x large positive", m, x);
  }
  printf("%d checks, %d failures, tolerance %.0e of the row bound\n", checks, failures, kTol);
  return failures != 0;
}
