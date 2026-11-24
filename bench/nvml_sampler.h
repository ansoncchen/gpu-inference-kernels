// Samples clocks, temperature, power and throttle reasons while a benchmark
// runs. NVML is loaded with dlopen so no NVML headers are needed.
#pragma once
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

class NvmlSampler {
 public:
  explicit NvmlSampler(const char* pci_bus_id) {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    auto init = (int (*)())dlsym(h, "nvmlInit_v2");
    auto get = (int (*)(const char*, void**))dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    clock_ = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetClockInfo");
    temp_ = (int (*)(void*, int, unsigned*))dlsym(h, "nvmlDeviceGetTemperature");
    power_ = (int (*)(void*, unsigned*))dlsym(h, "nvmlDeviceGetPowerUsage");
    throttle_ = (int (*)(void*, unsigned long long*))dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
    ok_ = init && get && clock_ && temp_ && power_ && init() == 0 && get(pci_bus_id, &dev_) == 0;
  }
  void start() {
    sm_.clear(), mem_.clear(), temp_s_.clear(), watts_.clear(), reasons_ = 0;
    if (!ok_) return;
    run_ = true;
    th_ = std::thread([this] {
      while (run_) {
        unsigned v;
        unsigned long long t;
        if (clock_(dev_, 1, &v) == 0) sm_.push_back(v);   // SM clock, MHz
        if (clock_(dev_, 2, &v) == 0) mem_.push_back(v);  // memory clock, MHz
        if (temp_(dev_, 0, &v) == 0) temp_s_.push_back(v);
        if (power_(dev_, &v) == 0) watts_.push_back(v / 1000.0);
        if (throttle_ && throttle_(dev_, &t) == 0) reasons_ |= t;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }
  std::string stop_json() {
    if (!ok_) return "null";
    run_ = false;
    th_.join();
    char b[400];
    snprintf(b, sizeof b,
             "{\"samples\":%zu,\"sm_mhz_min\":%.0f,\"sm_mhz_mean\":%.0f,\"sm_mhz_max\":%.0f,\"mem_mhz_mean\":%.0f,"
             "\"temp_c_max\":%.0f,\"power_w_mean\":%.1f,\"throttle_mask\":%llu}",
             sm_.size(), min(sm_), mean(sm_), max(sm_), mean(mem_), max(temp_s_), mean(watts_), reasons_);
    return b;
  }

 private:
  using V = std::vector<double>;
  static double min(const V& v) { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
  static double max(const V& v) { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }
  static double mean(const V& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
  }
  bool ok_ = false;
  void* dev_ = nullptr;
  int (*clock_)(void*, int, unsigned*) = nullptr;
  int (*temp_)(void*, int, unsigned*) = nullptr;
  int (*power_)(void*, unsigned*) = nullptr;
  int (*throttle_)(void*, unsigned long long*) = nullptr;
  V sm_, mem_, temp_s_, watts_;
  unsigned long long reasons_ = 0;
  std::atomic<bool> run_{false};
  std::thread th_;
};
