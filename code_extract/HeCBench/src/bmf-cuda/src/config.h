#ifndef CONFIG_H
#define CONFIG_H

#include <cstdlib>
#include <cstdint>
#include <limits>

#ifndef WARPSPERBLOCK
#define WARPSPERBLOCK 16
#endif
#ifndef INITIALIZATIONMODE
#define INITIALIZATIONMODE 1
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

template <typename index_t, typename error_t>
struct cuBool_config {
  size_t verbosity = 1;
  index_t linesAtOnce = 0;
  size_t maxIterations = 0;
  error_t distanceThreshold = std::numeric_limits<error_t>::lowest();
  size_t distanceShowEvery = std::numeric_limits<size_t>::max();
  float tempStart = 0.0f;
  float tempEnd = -1.0f;
  float reduceFactor = 0.98f;
  size_t reduceStep = std::numeric_limits<size_t>::max();
  uint32_t seed = 0;
  bool loadBalance = false;
  float flipManyChance = 0.1f;
  uint32_t flipManyDepth = 2;
  size_t stuckIterationsBeforeBreak = std::numeric_limits<size_t>::max();
  uint8_t factorDim = 20;
  int weight = 1;
};

#endif
