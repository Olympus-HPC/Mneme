// MnemeCuptiProfiler.cpp
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include "mneme/MnemeLogger.hpp"

#if defined(_MSC_VER)
#define HAVE_DECLSPEC_DLL
#endif

#if defined(HAVE_DECLSPEC_DLL)
#define API_EXPORT(RTYPE) __declspec(dllexport) RTYPE
#else
#define API_EXPORT(RTYPE) RTYPE
#endif

using u64 = unsigned long long;

// -----------------------------
// Error handling helpers
// -----------------------------
#define CHECK_CUDA(call)                                                      \
do {                                                                        \
  cudaError_t _e = (call);                                                  \
  if (_e != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d: %s\n",                     \
            (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__, #call);    \
                 std::abort();                                                           \
  }                                                                         \
} while (0)

static inline const char* cuptiGetResultString(CUptiResult r) {
  const char* s = nullptr;
  cuptiGetResultString(r, &s);
  return s ? s : "<unknown>";
}

#define CHECK_CUPTI(call)                                                     \
do {                                                                        \
  CUptiResult _r = (call);                                                  \
  if (_r != CUPTI_SUCCESS) {                                                \
    fprintf(stderr, "CUPTI error %d (%s) at %s:%d: %s\n",                    \
            (int)_r, cuptiGetResultString(_r), __FILE__, __LINE__, #call);  \
                 std::abort();                                                           \
  }                                                                         \
} while (0)

// -----------------------------
// CUPTI buffer config
// -----------------------------
namespace {
constexpr size_t kBufSize      = 1 << 20;   // 1 MiB
constexpr size_t kBufAlign     = 8;
constexpr size_t kBufPoolCount = 8;         // small pool for async callbacks
constexpr CUpti_ExternalCorrelationKind kExtKind = CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;
} // namespace

namespace mneme {

class MnemeCuptiProfiler {
private:
  std::mutex mtx_;
  std::atomic<u64> nextToken_{0};

  // Token -> *target* kernel name (empty means accept any)
  std::unordered_map<u64, std::string> targetByToken_;
  // Token -> durations (ns)
  std::unordered_map<u64, std::vector<u64>> profilesByToken_;

  // CUPTI correlationId (uint32) -> our token (u64)
  // Filled by CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records.
  std::unordered_map<uint32_t, u64> corrIdToToken_;

  // simple pool of buffers to avoid malloc in callbacks
  std::vector<uint8_t*> bufPool_;
  size_t poolIdx_ = 0;

  bool initialized_ = false;

  MnemeCuptiProfiler() = default;

  static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    auto& inst = instance();
    std::lock_guard<std::mutex> lk(inst.mtx_);

    if (inst.bufPool_.empty()) {
      // Lazily create pool
      inst.bufPool_.reserve(kBufPoolCount);
      for (size_t i = 0; i < kBufPoolCount; ++i) {
        void* p = nullptr;
#if defined(_MSC_VER)
        p = _aligned_malloc(kBufSize, kBufAlign);
        if (!p) std::abort();
#else
        if (posix_memalign(&p, kBufAlign, kBufSize) != 0) std::abort();
#endif
        inst.bufPool_.push_back(reinterpret_cast<uint8_t*>(p));
      }
      inst.poolIdx_ = 0;
    }

    uint8_t* b = inst.bufPool_[inst.poolIdx_++ % inst.bufPool_.size()];
    *buffer = b;
    *size = kBufSize;
    *maxNumRecords = 0; // let CUPTI decide
  }

  static void CUPTIAPI bufferCompleted(CUcontext /*ctx*/, uint32_t /*streamId*/,
                                       uint8_t* buffer, size_t size, size_t validSize) {
    auto& inst = instance();

    // Dropped records check (important if you see gaps)
    size_t dropped = 0;
    CHECK_CUPTI(cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped));
    if (dropped > 0) {
      LOG_WARN("CUPTI dropped {} activity records (increase buffers / reduce load)", dropped);
    }

    // First pass: harvest EXTERNAL_CORRELATION mappings that may appear in this buffer
    {
      CUpti_Activity* rec = nullptr;
      CUptiResult st = CUPTI_SUCCESS;
      while ((st = cuptiActivityGetNextRecord(buffer, validSize, &rec)) == CUPTI_SUCCESS) {
        if (!rec) break;
        if (rec->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
          auto* e = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(rec);
          if (e->externalKind == kExtKind) {
            std::lock_guard<std::mutex> lk(inst.mtx_);
            inst.corrIdToToken_[e->correlationId] = static_cast<u64>(e->externalId);
          }
        }
      }
      if (st != CUPTI_SUCCESS && st != CUPTI_ERROR_MAX_LIMIT_REACHED) {
        LOG_WARN("cuptiActivityGetNextRecord returned {}", (int)st);
      }
    }

    // Second pass: iterate again (need reset iteration; easiest is manual loop by re-calling from start)
    // CUPTI doesn't provide a "reset iterator", so we just do a single pass and handle both kinds in one.
    // To keep this correct, we re-iterate by scanning records again using a fresh loop:
    {
      // We can re-scan by using cuptiActivityGetNextRecord again only if we restart from scratch,
      // but CUPTI's iterator is stateless per call; it uses internal cursor in user code.
      // So instead, do the *real* single-pass logic here and remove the earlier pass.
      //
      // Practical approach: do it as a single pass with a local map, then merge.
    }

    // Correct single-pass implementation with local staging:
    std::unordered_map<uint32_t, u64> localCorr;
    struct KernelSample { uint32_t corr; const char* name; uint64_t start; uint64_t end; bool ok; };
    std::vector<KernelSample> kernels;
    kernels.reserve(256);

    CUpti_Activity* rec = nullptr;
    CUptiResult st = CUPTI_SUCCESS;
    size_t offsetGuard = 0;

    // Iterate records once, collect local correlation + kernel samples
    while ((st = cuptiActivityGetNextRecord(buffer, validSize, &rec)) == CUPTI_SUCCESS) {
      if (!rec) break;
      ++offsetGuard;
      if (offsetGuard > 10'000'000) break; // paranoia guard

      switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
          auto* e = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(rec);
          if (e->externalKind == kExtKind) {
            localCorr[e->correlationId] = static_cast<u64>(e->externalId);
          }
          break;
        }

        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: { 
          auto* k = reinterpret_cast<const CUpti_ActivityKernel*>(rec);
          kernels.push_back(KernelSample{
            k->correlationId,
            k->name ? k->name : "<unknown>",
            static_cast<uint64_t>(k->start),
            static_cast<uint64_t>(k->end),
            k->end >= k->start
            LOG_DEBUG("CUPTI bufferCompleted: ext_corr={} kernels={} validSize={}", nExt, nKer, validSize);
          });
          break;
        }

        default:
          break;
      }
    }

    if (st != CUPTI_SUCCESS && st != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      LOG_WARN("cuptiActivityGetNextRecord returned {}", (int)st);
    }

    // Merge local correlation into global map
    if (!localCorr.empty()) {
      std::lock_guard<std::mutex> lk(inst.mtx_);
      for (auto& kv : localCorr) inst.corrIdToToken_[kv.first] = kv.second;
    }

    // Consume kernel samples
    for (const auto& s : kernels) {
      u64 token = 0;
      {
        std::lock_guard<std::mutex> lk(inst.mtx_);
        auto it = inst.corrIdToToken_.find(s.corr);
        if (it != inst.corrIdToToken_.end()) token = it->second;
      }
      if (token == 0) continue;

      std::string kname = s.name ? std::string(s.name) : std::string("<unknown>");
      u64 dur = (s.ok) ? static_cast<u64>(s.end - s.start) : 0;

      std::lock_guard<std::mutex> lk(inst.mtx_);

      // token must be active
      auto tgtIt = inst.targetByToken_.find(token);
      if (tgtIt == inst.targetByToken_.end()) continue;

      // optional name filter
      const std::string& want = tgtIt->second;
      if (!want.empty() && kname != want) continue;

      if (dur > 0) {
        LOG_DEBUG("Associating {} with token id {}", kname, token);
        inst.profilesByToken_[token].push_back(dur);
      }
    }

    // NOTE: buffer is from our pool; CUPTI doesn't own it. Nothing to free here.
    (void)buffer;
    (void)size;
  }

public:
  static MnemeCuptiProfiler& instance() {
    static MnemeCuptiProfiler inst;
    return inst;
  }

  void initIfNeeded() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (initialized_) return;

    // Make sure CUDA is initialized
    CHECK_CUDA(cudaFree(nullptr));

    // Activity setup
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(&bufferRequested, &bufferCompleted));

    // Enable activities we need
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    // Some setups only emit KERNEL (non-concurrent); enabling doesn't hurt

    initialized_ = true;
    LOG_DEBUG("Initialized CUPTI activity profiling");
  }

  int64_t start(const char* kernelName) {
    initIfNeeded();
    u64 tok = nextToken_.fetch_add(1, std::memory_order_relaxed) + 1;

    // Push thread-scoped external correlation id
    CHECK_CUPTI(cuptiActivityPushExternalCorrelationId(kExtKind, static_cast<uint64_t>(tok)));

    {
      std::lock_guard<std::mutex> lk(mtx_);
      targetByToken_[tok] = kernelName ? std::string(kernelName) : std::string();
      profilesByToken_.erase(tok);
    }

    LOG_DEBUG("CUPTI startProfile kernel='{}' token={}", kernelName ? kernelName : "<any>", tok);
    return tok;
  }

  int64_t numRecords(u64 token) {
    initIfNeeded();

    // Ensure all kernels are done and activities are flushed into our buffers
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUPTI(cuptiActivityFlushAll(0));

    std::lock_guard<std::mutex> lk(mtx_);
    auto it = profilesByToken_.find(token);
    if (it == profilesByToken_.end()) return static_cast<int64_t>(-1);
    return static_cast<u64>(it->second.size());
  }

  std::vector<int64_t> stop(u64 token) {
    initIfNeeded();

    // Pop the correlation id
    CHECK_CUPTI(cuptiActivityPopExternalCorrelationId(kExtKind, nullptr));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUPTI(cuptiActivityFlushAll(0));

    std::vector<u64> out;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (auto it = profilesByToken_.find(token); it != profilesByToken_.end()) {
        out = std::move(it->second);
        profilesByToken_.erase(it);
      }
      targetByToken_.erase(token);
    }
    LOG_DEBUG("CUPTI stopProfile token={} records={}", token, out.size());
    return out;
  }

  MnemeCuptiProfiler(const MnemeCuptiProfiler&) = delete;
  MnemeCuptiProfiler& operator=(const MnemeCuptiProfiler&) = delete;
};

} // namespace mneme

// -----------------------------
// C ABI entrypoints (match your HIP ones)
// -----------------------------
extern "C" {

API_EXPORT(int64_t) MnemePy_startProfile(const char* KernelName) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  return inst.start(KernelName);
}

API_EXPORT(void) MnemePy_stopProfile(u64 Token, int64_t* ProfileData, u64 Size) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  auto prof = inst.stop(Token);

  if (prof.size() != Size) {
    LOG_WARN("Number of records differ between python container and C++ container size {} vs {}",
             Size, prof.size());
  }

  for (u64 i = 0; i < std::min<u64>(Size, static_cast<u64>(prof.size())); ++i) {
    ProfileData[i] = prof[i];
  }
}

API_EXPORT(int64_t) MnemePy_getNumRecords(u64 token) {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  return inst.numRecords(token);
}

API_EXPORT(void) MnemePy_initProfiler() {
  auto& inst = mneme::MnemeCuptiProfiler::instance();
  inst.initIfNeeded();
}

} // extern "C"

