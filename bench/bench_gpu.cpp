// GPU benchmark harness. One JSON object per line on --out (appended).
//
// Method: every rep runs one GEMV on each of K separate copies of the matrix,
// where K copies together hold at least --footprint MB, so nothing is reused
// from the 5.5 MB L2 (decode walks through many layers in the same way). The
// launches are queued back to back and timed on the GPU with events; the value
// recorded is (time for the pass) / K. Launch gaps are therefore included. An
// empty-kernel row reports their size. Single-call latency (launch + sync, host
// clock) is recorded separately.
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <sstream>

#include "gpu.h"
#include "nvml_sampler.h"
#include "stats.h"

using namespace q4;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

#define CUBLAS_CHECK(e)                                                        \
  do {                                                                         \
    cublasStatus_t s_ = (e);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct Options {
  std::vector<std::pair<int, int>> shapes{{4096, 4096}, {11008, 4096}, {8192, 8192}, {32768, 8192}};
  std::set<std::string> only;
  std::string out = "results/raw/gpu.jsonl", label = "steps";
  double footprint_mb = 1024, min_s = 0.5, max_s = 4.0, warm_s = 0.25;
};

struct Timing {
  std::vector<double> pass_us;  // per GEMV
  std::vector<double> latency_us;
  std::string nvml;
};

static cudaStream_t g_stream;
static NvmlSampler* g_nvml;

// `pass` queues the K launches; `single` queues one.
static Timing measure(const Options& o, int k, const std::function<void()>& pass, const std::function<void()>& single) {
  Timing t;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  auto w0 = Clock::now();
  do {
    pass();
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
  } while (secs(w0, Clock::now()) < o.warm_s);
  g_nvml->start();
  auto b0 = Clock::now();
  while (true) {
    CUDA_CHECK(cudaEventRecord(e0, g_stream));
    pass();
    CUDA_CHECK(cudaEventRecord(e1, g_stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    t.pass_us.push_back(ms * 1e3 / k);
    double s = secs(b0, Clock::now());
    if ((s > o.min_s && t.pass_us.size() >= 30) || (s > o.max_s && t.pass_us.size() >= 5) || t.pass_us.size() >= 2000) break;
  }
  t.nvml = g_nvml->stop_json();
  for (int i = 0; i < 200; i++) {
    auto a = Clock::now();
    single();
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    t.latency_us.push_back(secs(a, Clock::now()) * 1e6);
    if (secs(b0, Clock::now()) > o.max_s + 1 && i >= 10) break;
  }
  CUDA_CHECK(cudaGetLastError());
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return t;
}

static void emit(const Options& o, const std::string& name, int rows, int cols, size_t bytes, int copies,
                 const Timing& t, double err, const char* extra = "") {
  std::ofstream f(o.out, std::ios::app);
  double med = quantile(t.pass_us, 0.5);
  f << "{\"run\":\"" << o.label << "\",\"variant\":\"" << name << "\",\"rows\":" << rows << ",\"cols\":" << cols
    << ",\"bytes\":" << bytes << ",\"copies\":" << copies << "," << stats_json(t.pass_us)
    << ",\"gbps_median\":" << bytes / med * 1e-3 << ",\"latency_median_us\":" << quantile(t.latency_us, 0.5)
    << ",\"max_rel_err\":" << err << ",\"nvml\":" << t.nvml << extra << "}\n";
  printf("%-22s %6dx%-6d %9.2f us  [p5 %.2f p95 %.2f]  %7.1f GB/s  lat %.1f us  err %.1e\n", name.c_str(), rows, cols,
         med, quantile(t.pass_us, 0.05), quantile(t.pass_us, 0.95), bytes / med * 1e-3, quantile(t.latency_us, 0.5), err);
  fflush(stdout);
}

static double max_err(const std::vector<float>& y, const std::vector<double>& ref, const std::vector<double>& bound) {
  double worst = 0;
  for (size_t r = 0; r < ref.size(); r++) {
    double e = std::fabs(y[r] - ref[r]) / bound[r];
    if (!(e <= worst)) worst = e;
  }
  return worst;
}

template <class T>
static T* dmalloc(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
template <class T>
static void upload(T* d, const T* h, size_t n) {
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    std::stringstream ss(v);
    std::string tok;
    if (k == "--shapes") {
      o.shapes.clear();
      while (std::getline(ss, tok, ',')) {
        int r, c;
        if (sscanf(tok.c_str(), "%dx%d", &r, &c) == 2) o.shapes.push_back({r, c});
      }
    } else if (k == "--variants") {
      while (std::getline(ss, tok, ',')) o.only.insert(tok);
    } else if (k == "--out") o.out = v;
    else if (k == "--label") o.label = v;
    else if (k == "--footprint") o.footprint_mb = atof(v.c_str());
    else if (k == "--max-s") o.max_s = atof(v.c_str());
  }
  auto want = [&](const std::string& n) {
    if (o.only.empty()) return n[0] != 't';  // tuning grid only on request
    for (const auto& s : o.only)
      if (n == s || (s.back() == '*' && n.compare(0, s.size() - 1, s, 0, s.size() - 1) == 0)) return true;
    return false;
  };

  char bus[32];
  CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof bus, 0));
  NvmlSampler nvml(bus);
  g_nvml = &nvml;
  CUDA_CHECK(cudaStreamCreate(&g_stream));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("%s, %d SMs, %.1f GB\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

  // --- probes ---------------------------------------------------------------
  uint32_t* sink = dmalloc<uint32_t>(4);
  if (want("empty")) {
    Timing t = measure(o, 1000, [&] { for (int i = 0; i < 1000; i++) launch_empty(g_stream); }, [&] { launch_empty(g_stream); });
    emit(o, "empty", 0, 0, 0, 1000, t, 0);
  }
  if (want("fma")) {
    double flops = 0;
    Timing t = measure(o, 1, [&] { flops = launch_fma(g_stream, (float*)sink); }, [&] { launch_fma(g_stream, (float*)sink); });
    char ex[64];
    snprintf(ex, sizeof ex, ",\"fma_per_s\":%.6e", flops / (quantile(t.pass_us, 0.5) * 1e-6));
    emit(o, "fma", 0, 0, 0, 1, t, 0, ex);
    printf("  FMA rate %.3e /s\n", flops / (quantile(t.pass_us, 0.5) * 1e-6));
  }
  for (int vec : {1, 4, 8, 16}) {
    std::string name = "read" + std::to_string(vec);
    if (!want(name)) continue;
    const size_t bytes = size_t(4) << 30;
    uint8_t* buf = dmalloc<uint8_t>(bytes);
    CUDA_CHECK(cudaMemset(buf, 0, bytes));
    Timing t = measure(o, 1, [&] { launch_read(buf, bytes, vec, sink, g_stream); }, [&] { launch_read(buf, bytes, vec, sink, g_stream); });
    emit(o, name, 0, 0, bytes, 1, t, 0);
    cudaFree(buf);
  }

  // --- GEMV -----------------------------------------------------------------
  for (auto [R, C] : o.shapes) {
    Matrix m = random_matrix(R, C, 42);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> x((C + 31) / 32 * 32);
    for (auto& v : x) v = nd(rng);
    std::vector<double> ref(R), bound(R);
    gemv_reference(m, x.data(), ref.data(), bound.data());
    int ypad = (R + 31) / 32 * 32;
    float* dx = dmalloc<float>(x.size());
    float* dy = dmalloc<float>(ypad);
    float* scratch = dmalloc<float>(2 * C + 2048);
    upload(dx, x.data(), x.size());
    std::vector<float> y(R);
    auto copies_for = [&](size_t b) { return std::max<int>(1, (int)std::ceil(o.footprint_mb * 1048576.0 / b)); };

    for (int tiled = 0; tiled < 2; tiled++) {
      bool any = false;
      for (const Variant& v : variants()) any |= v.tiled == (bool)tiled && want(v.name);
      if (!any) continue;
      Tiled t;
      if (tiled) t = to_tiled(m);
      const std::vector<uint8_t>& hw = tiled ? t.w : m.w;
      const std::vector<uint16_t>& hs = tiled ? t.scale : m.scale;
      size_t bytes = hw.size() + hs.size() * 2;
      int K = copies_for(bytes);
      size_t wstep = (hw.size() + 255) / 256 * 256, sstep = (hs.size() * 2 + 255) / 256 * 256;
      uint8_t* dw = dmalloc<uint8_t>(wstep * K);
      uint8_t* ds = dmalloc<uint8_t>(sstep * K);
      std::vector<GpuMat> g(K);
      for (int i = 0; i < K; i++) {
        upload(dw + wstep * i, hw.data(), hw.size());
        upload((uint16_t*)(ds + sstep * i), hs.data(), hs.size());
        g[i].rows = R, g[i].cols = C, g[i].tile_rows = (R + 31) / 32, g[i].tile_cols = (C + 1023) / 1024;
        (tiled ? g[i].tw : g[i].w) = dw + wstep * i;
        (tiled ? g[i].tscale : g[i].scale) = (uint16_t*)(ds + sstep * i);
      }
      // Same bytes, same number of launches, but only reads: the best any kernel
      // could do at this matrix size on this machine, launch gaps included.
      if (tiled && want("readbound")) {
        Timing tm = measure(o, K, [&] { for (int i = 0; i < K; i++) launch_read(dw + wstep * i, hw.size(), 16, sink, g_stream); },
                            [&] { launch_read(dw, hw.size(), 16, sink, g_stream); });
        emit(o, "readbound", R, C, hw.size() / 32768 * 32768, K, tm, 0);
      }
      for (const Variant& v : variants()) {
        if (v.tiled != (bool)tiled || !want(v.name)) continue;
        Timing tm = measure(o, K, [&] { for (auto& a : g) v.launch(a, dx, dy, scratch, g_stream); },
                            [&] { v.launch(g[0], dx, dy, scratch, g_stream); });
        CUDA_CHECK(cudaMemcpy(y.data(), dy, R * 4, cudaMemcpyDeviceToHost));
        emit(o, v.name, R, C, bytes, K, tm, max_err(y, ref, bound));
      }
      cudaFree(dw);
      cudaFree(ds);
    }

    // cuBLAS baselines on the same (dequantized) weights.
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetStream(h, g_stream));
    std::vector<float> a;
    auto need = [&] { if (a.empty()) a = dequantize(m); };
    const float one = 1, zero = 0;
    for (bool trans : {true, false}) {
      // trans: A stored row by row (one dot product per output).
      // no trans: A stored column by column.
      std::string sfx = trans ? "_T" : "_N";
      auto store = [&](auto conv) {
        using T = decltype(conv(0.0f));
        std::vector<T> s((size_t)R * C);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < R; r++)
          for (int c = 0; c < C; c++) s[trans ? (size_t)r * C + c : (size_t)c * R + r] = conv(a[(size_t)r * C + c]);
        return s;
      };
      int M = trans ? C : R, N = trans ? R : C;
      cublasOperation_t op = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
      if (want("fp32_sgemv" + sfx)) {
        need();
        auto s = store([](float f) { return f; });
        size_t n = s.size(), bytes = n * 4;
        int K = copies_for(bytes);
        float* d = dmalloc<float>(n * K);
        for (int i = 0; i < K; i++) upload(d + n * i, s.data(), n);
        auto one_call = [&](int i) { CUBLAS_CHECK(cublasSgemv(h, op, M, N, &one, d + n * i, M, dx, 1, &zero, dy, 1)); };
        Timing tm = measure(o, K, [&] { for (int i = 0; i < K; i++) one_call(i); }, [&] { one_call(0); });
        CUDA_CHECK(cudaMemcpy(y.data(), dy, R * 4, cudaMemcpyDeviceToHost));
        emit(o, "fp32_sgemv" + sfx, R, C, bytes, K, tm, max_err(y, ref, bound));
        cudaFree(d);
      }
      bool w1 = want("fp16_HSHgemv" + sfx), w2 = want("fp16_HSSgemv" + sfx), w3 = want("fp16_gemmEx" + sfx);
      if (!w1 && !w2 && !w3) continue;
      need();
      auto s = store([](float f) { return float_to_half(f); });
      size_t n = s.size(), bytes = n * 2;
      int K = copies_for(bytes);
      uint16_t* d = dmalloc<uint16_t>(n * K);
      for (int i = 0; i < K; i++) upload(d + n * i, s.data(), n);
      std::vector<uint16_t> hx(x.size()), hy(R);
      for (size_t i = 0; i < x.size(); i++) hx[i] = float_to_half(x[i]);
      uint16_t* dhx = dmalloc<uint16_t>(hx.size());
      uint16_t* dhy = dmalloc<uint16_t>(R);
      upload(dhx, hx.data(), hx.size());
      auto half_out = [&] {
        CUDA_CHECK(cudaMemcpy(hy.data(), dhy, R * 2, cudaMemcpyDeviceToHost));
        for (int r = 0; r < R; r++) y[r] = half_to_float(hy[r]);
      };
      if (w1) {
        auto f = [&](int i) {
          CUBLAS_CHECK(cublasHSHgemvStridedBatched(h, op, M, N, &one, (__half*)(d + n * i), M, n, (__half*)dhx, 1, C, &zero, (__half*)dhy, 1, R, 1));
        };
        Timing tm = measure(o, K, [&] { for (int i = 0; i < K; i++) f(i); }, [&] { f(0); });
        half_out();
        emit(o, "fp16_HSHgemv" + sfx, R, C, bytes, K, tm, max_err(y, ref, bound));
      }
      if (w2) {
        auto f = [&](int i) {
          CUBLAS_CHECK(cublasHSSgemvStridedBatched(h, op, M, N, &one, (__half*)(d + n * i), M, n, (__half*)dhx, 1, C, &zero, dy, 1, R, 1));
        };
        Timing tm = measure(o, K, [&] { for (int i = 0; i < K; i++) f(i); }, [&] { f(0); });
        CUDA_CHECK(cudaMemcpy(y.data(), dy, R * 4, cudaMemcpyDeviceToHost));
        emit(o, "fp16_HSSgemv" + sfx, R, C, bytes, K, tm, max_err(y, ref, bound));
      }
      if (w3) {
        // y (R x 1) = op(A) x (C x 1)
        auto f = [&](int i) {
          CUBLAS_CHECK(cublasGemmEx(h, op, CUBLAS_OP_N, R, 1, C, &one, d + n * i, CUDA_R_16F, M, dhx, CUDA_R_16F, C,
                                    &zero, dhy, CUDA_R_16F, R, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        };
        Timing tm = measure(o, K, [&] { for (int i = 0; i < K; i++) f(i); }, [&] { f(0); });
        half_out();
        emit(o, "fp16_gemmEx" + sfx, R, C, bytes, K, tm, max_err(y, ref, bound));
      }
      cudaFree(d);
      cudaFree(dhx);
      cudaFree(dhy);
    }
    cublasDestroy(h);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(scratch);
  }
  return 0;
}
