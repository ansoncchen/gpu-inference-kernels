// llama.cpp / ggml CUDA baseline: Q4_0 mul_mat with one vector, through the
// public ggml API. Same weights (bit for bit the same nibbles and FP16 scales),
// same x, same shapes, same K-copies method as bench_gpu.cpp.
//
// Differences to keep in mind: ggml quantizes x to int8 per 32-value block on
// the GPU and does integer dot products, so its result carries an extra error
// that our FP32-x kernels do not have. Timing is host-side around a whole graph
// of K mul_mats plus synchronize; ggml may capture that graph as a CUDA graph.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>

#include "../bench/nvml_sampler.h"
#include "../bench/stats.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "q4.h"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
  const char* out = argc > 1 ? argv[1] : "results/raw/ggml.jsonl";
  const char* bus = argc > 2 ? argv[2] : "0000:00:00.0";
  int shapes[][2] = {{1536, 8960}, {8960, 1536}, {4096, 4096}, {11008, 4096}, {8192, 8192}, {32768, 8192}};
  NvmlSampler nvml(bus);
  ggml_backend_t be = ggml_backend_cuda_init(0);
  if (!be) return 1;
  for (auto& s : shapes) {
    int R = s[0], C = s[1];
    q4::Matrix m = q4::random_matrix(R, C, 42);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> x(C);
    for (auto& v : x) v = nd(rng);
    std::vector<double> ref(R), bound(R);
    q4::gemv_reference(m, x.data(), ref.data(), bound.data());

    // Q4_0 block: FP16 scale, then 16 bytes with nibble j low and j+16 high.
    std::vector<uint8_t> blocks(m.bytes());
    int G = m.groups();
#pragma omp parallel for schedule(static)
    for (int r = 0; r < R; r++)
      for (int g = 0; g < G; g++) {
        uint8_t* b = &blocks[((size_t)r * G + g) * 18];
        memcpy(b, &m.scale[(size_t)r * G + g], 2);
        for (int j = 0; j < 16; j++)
          b[2 + j] = uint8_t(m.q(r, g * 32 + j) + 8) | uint8_t(m.q(r, g * 32 + j + 16) + 8) << 4;
      }

    size_t bytes = blocks.size();
    int K = std::max<int>(1, (int)std::ceil(1024.0 * 1048576.0 / bytes));
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(K * 3 + 16) + ggml_graph_overhead_custom(K + 8, false), nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    std::vector<ggml_tensor*> w(K);
    for (auto& t : w) t = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_0, C, R);
    ggml_tensor* tx = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, C, 1);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto* t : w) ggml_backend_tensor_set(t, blocks.data(), 0, bytes);
    ggml_backend_tensor_set(tx, x.data(), 0, C * 4);

    ggml_context* gctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(gctx, K + 8, false);
    ggml_tensor* last = nullptr;
    std::vector<ggml_tensor*> outs;
    for (auto* t : w) {
      last = ggml_mul_mat(gctx, t, tx);
      ggml_set_output(last);
      ggml_build_forward_expand(gf, last);
    }
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be);

    auto run = [&] {
      ggml_backend_graph_compute_async(be, gf);
      ggml_backend_synchronize(be);
    };
    auto w0 = Clock::now();
    do run();
    while (secs(w0, Clock::now()) < 0.3);
    std::vector<double> us;
    nvml.start();
    auto b0 = Clock::now();
    while (us.size() < 30 || (secs(b0, Clock::now()) < 0.5 && us.size() < 2000)) {
      auto a = Clock::now();
      run();
      us.push_back(secs(a, Clock::now()) * 1e6 / K);
    }
    std::string nv = nvml.stop_json();
    std::vector<float> y(R);
    ggml_backend_tensor_get(last, y.data(), 0, R * 4);
    double worst = 0;
    for (int r = 0; r < R; r++) {
      double e = std::fabs(y[r] - ref[r]) / bound[r];
      if (!(e <= worst)) worst = e;
    }
    double med = quantile(us, 0.5);
    std::ofstream f(out, std::ios::app);
    f << "{\"run\":\"ggml\",\"variant\":\"ggml_q4_0\",\"rows\":" << R << ",\"cols\":" << C << ",\"bytes\":" << bytes
      << ",\"copies\":" << K << "," << stats_json(us) << ",\"gbps_median\":" << bytes / med * 1e-3
      << ",\"latency_median_us\":null,\"max_rel_err\":" << worst << ",\"nvml\":" << nv << "}\n";
    printf("ggml_q4_0 %6dx%-6d %9.2f us  %7.1f GB/s  err %.1e\n", R, C, med, bytes / med * 1e-3, worst);
    ggml_free(gctx);
    ggml_free(wctx);
    ggml_backend_buffer_free(gbuf);
    ggml_backend_buffer_free(wbuf);
  }
  ggml_backend_free(be);
}
