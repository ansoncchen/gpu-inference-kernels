// Minimal reader for a single .safetensors file (BF16 / F16 / F32 tensors).
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "q4.h"

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd, &st);
    base_ = (const char*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t hlen;
    memcpy(&hlen, base_, 8);
    std::string h(base_ + 8, hlen);
    data_ = base_ + 8 + hlen;
    // Entries look like "name":{"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]}
    size_t p = 0;
    while ((p = h.find("{\"dtype\":\"", p)) != std::string::npos) {
      size_t q1 = h.rfind('"', p), q0 = h.rfind('"', q1 - 1);
      Entry e;
      size_t d0 = p + 10, d1 = h.find('"', d0);
      e.dtype = h.substr(d0, d1 - d0);
      size_t s0 = h.find("\"shape\":[", p) + 9, s1 = h.find(']', s0);
      for (size_t i = s0; i < s1;) {
        e.shape.push_back(atol(h.c_str() + i));
        size_t c = h.find(',', i);
        i = (c == std::string::npos || c > s1) ? s1 : c + 1;
      }
      size_t o0 = h.find("\"data_offsets\":[", p) + 16;
      e.off = strtoull(h.c_str() + o0, nullptr, 10);
      map_[h.substr(q0 + 1, q1 - q0 - 1)] = e;
      p = o0;
    }
  }
  std::vector<float> get(const std::string& name) const {
    auto it = map_.find(name);
    if (it == map_.end()) throw std::runtime_error("missing tensor " + name);
    const Entry& e = it->second;
    size_t n = 1;
    for (long d : e.shape) n *= d;
    std::vector<float> out(n);
    const char* src = data_ + e.off;
    if (e.dtype == "F32") {
      memcpy(out.data(), src, n * 4);
    } else {
      const bool bf = e.dtype == "BF16";
      if (!bf && e.dtype != "F16") throw std::runtime_error("unsupported dtype " + e.dtype);
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, src + i * 2, 2);
        if (bf) {
          uint32_t b = (uint32_t)h << 16;
          memcpy(&out[i], &b, 4);
        } else {
          out[i] = q4::half_to_float(h);
        }
      }
    }
    return out;
  }

 private:
  struct Entry {
    std::string dtype;
    std::vector<long> shape;
    size_t off = 0;
  };
  const char* base_;
  const char* data_;
  std::map<std::string, Entry> map_;
};
