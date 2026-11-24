// GPU kernel variants. Every variant computes y = W x with FP32 x and y.
#pragma once
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "q4.h"

#define CUDA_CHECK(e)                                                              \
  do {                                                                             \
    cudaError_t err_ = (e);                                                        \
    if (err_ != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

namespace q4 {

// Device pointers for one matrix. Row-major and tiled copies are separate; a
// benchmark only uploads the one its variant reads.
struct GpuMat {
  int rows = 0, cols = 0, tile_rows = 0, tile_cols = 0;
  const uint8_t* w = nullptr;       // row-major nibbles
  const uint16_t* scale = nullptr;  // row-major FP16 scales
  const uint8_t* tw = nullptr;      // tiled nibbles
  const uint16_t* tscale = nullptr;
  int octets = 0, chunks = 0;
  const uint8_t* ow = nullptr;       // octet layout
  const uint16_t* oscale = nullptr;
};

enum Layout { kRowMajor = 0, kTiled = 1, kOctet = 2 };

struct Variant {
  std::string name;
  Layout layout;
  // Device buffer sizes in floats: y round_up(rows, 32), x cols, scratch
  // 2 * cols + 2048.
  void (*launch)(const GpuMat&, const float* x, float* y, float* scratch, cudaStream_t);
  const char* what;  // one-line description
};

const std::vector<Variant>& variants();

// Peak-read probe: every thread reads VecBytes-wide words of a large buffer.
void launch_read(const void* buf, size_t bytes, int vec_bytes, uint32_t* sink, cudaStream_t);
void launch_empty(cudaStream_t);
// Pure arithmetic probe: number of chained FP32 FMAs per thread is fixed; used
// only to report the measured FMA rate.
double launch_fma(cudaStream_t, float* sink);  // returns number of FMAs issued

}  // namespace q4
