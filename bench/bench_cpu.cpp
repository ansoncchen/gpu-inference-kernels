// CPU benchmark: same K-copies method as bench_gpu (at least 512 MB of matrix
// data per pass, so nothing comes from the 32 MB L3).
// Usage: bench_cpu out.jsonl [threads,threads,...]
#include <cblas.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>

#include "cpu.h"
#include "stats.h"

extern "C" void openblas_set_num_threads(int);

using namespace q4;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static std::vector<double> measure(int k, const std::function<void()>& pass) {
  auto w0 = Clock::now();
  do pass();
  while (secs(w0, Clock::now()) < 0.3);
  std::vector<double> us;
  auto b0 = Clock::now();
  while (us.size() < 20 || (secs(b0, Clock::now()) < 1.0 && us.size() < 500)) {
    auto a = Clock::now();
    pass();
    us.push_back(secs(a, Clock::now()) * 1e6 / k);
    if (secs(b0, Clock::now()) > 20 && us.size() >= 5) break;
  }
  return us;
}

static double loadavg() {
  double l = -1;
  std::ifstream("/proc/loadavg") >> l;
  return l;
}

int main(int argc, char** argv) {
  const char* out = argc > 1 ? argv[1] : "results/raw/cpu.jsonl";
  std::vector<int> threads;
  std::stringstream ss(argc > 2 ? argv[2] : "1,16,32");
  for (std::string t; std::getline(ss, t, ',');) threads.push_back(atoi(t.c_str()));
  double load0 = loadavg();
  auto emit = [&](const std::string& name, int th, int R, int C, size_t bytes, int K, const std::vector<double>& us, double err) {
    double med = quantile(us, 0.5);
    std::ofstream f(out, std::ios::app);
    f << "{\"run\":\"cpu\",\"variant\":\"" << name << "\",\"threads\":" << th << ",\"rows\":" << R << ",\"cols\":" << C
      << ",\"bytes\":" << bytes << ",\"copies\":" << K << "," << stats_json(us) << ",\"gbps_median\":" << bytes / med * 1e-3
      << ",\"max_rel_err\":" << err << ",\"loadavg_at_start\":" << load0 << "}\n";
    printf("%-16s t=%-3d %6dx%-6d %10.1f us [p5 %.1f p95 %.1f] %6.1f GB/s err %.1e\n", name.c_str(), th, R, C, med,
           quantile(us, 0.05), quantile(us, 0.95), bytes / med * 1e-3, err);
    fflush(stdout);
  };

  {
    const size_t n = size_t(1) << 28;  // 1 GB of floats
    std::unique_ptr<float[]> buf(new float[n]);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) buf[i] = 1.0f;
    double keep = 0;
    for (int th : threads) {
      omp_set_num_threads(th);
      auto us = measure(1, [&] { keep += cpu_read(buf.get(), n); });
      emit("read", th, 0, 0, n * 4, 1, us, 0);
    }
    if (keep == 0.5) puts("");
  }

  int shapes[][2] = {{4096, 4096}, {11008, 4096}, {8192, 8192}};
  for (auto& s : shapes) {
    int R = s[0], C = s[1];
    omp_set_num_threads(16);
    Matrix m = random_matrix(R, C, 42);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> x(C), y(R);
    for (auto& v : x) v = nd(rng);
    std::vector<double> ref(R), bound(R);
    gemv_reference(m, x.data(), ref.data(), bound.data());
    auto err = [&] {
      double w = 0;
      for (int r = 0; r < R; r++) {
        double e = std::fabs(y[r] - ref[r]) / bound[r];
        if (!(e <= w)) w = e;
      }
      return w;
    };
    int K = (int)std::ceil(512.0 * 1048576.0 / m.bytes());
    std::vector<Matrix> ms(K, m);
    for (const auto& v : cpu_variants())
      for (int th : threads) {
        if (v.name == "c0_scalar" && th == 1) continue;  // minutes per point, not needed
        omp_set_num_threads(th);
        auto us = measure(K, [&] { for (auto& a : ms) v.run(a, x.data(), y.data()); });
        emit(v.name, th, R, C, m.bytes(), K, us, err());
      }
    ms.clear();
    size_t n = (size_t)R * C;
    int KF = (int)std::ceil(512.0 * 1048576.0 / (n * 4));
    auto d = dequantize(m);
    std::unique_ptr<float[]> fw(new float[n * KF]);
    for (int k = 0; k < KF; k++) memcpy(&fw[n * k], d.data(), n * 4);
    for (int th : threads) {
      openblas_set_num_threads(th);
      auto us = measure(KF, [&] {
        for (int k = 0; k < KF; k++)
          cblas_sgemv(CblasRowMajor, CblasNoTrans, R, C, 1.0f, &fw[n * k], C, x.data(), 1, 0.0f, y.data(), 1);
      });
      emit("openblas_fp32", th, R, C, n * 4, KF, us, err());
    }
  }
}
