#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "Gate.h"

#include <cuda_runtime.h>
#include <unordered_map>
using std::make_pair;
using std::unordered_map;

#define TRUTH_SIZE 1025
#define C_THREAD_LIMIT 32

const size_t C_PARALLEL_LIMIT = 10000;

typedef struct {
  size_t funcSer;
  Event *iPort[MAX_INPUT_PORT];
  dUnit dTable[MAX_DTABLE];
} gpu_GATE;

class Simulator {
private:
  tUnit dumpOff;

  size_t parallelSize;

  char *d_eTableMgr;
  gpu_GATE *d_gateMgr;

  bool h_overflow[C_PARALLEL_LIMIT];
  bool *d_overflow;

  size_t h_oHisSize[C_PARALLEL_LIMIT];
  size_t *d_oHisSize;

  size_t h_oHisMax[C_PARALLEL_LIMIT];
  size_t *d_oHisMax;

  Event *h_oHisArr[C_PARALLEL_LIMIT];
  Event **d_oHisArr;

  unordered_map<size_t, Event *> d_wireMapper;
  // Temp cache
  Event *d_HisTmp;

public:
  Simulator(tUnit);
  ~Simulator() {}

  void addEvalMgr(char *, size_t);
  void addTrans(tHistory *, size_t &);
  Event *getTrans(size_t &);
  void popTrans(size_t &);
  void addGateMgr(gpu_GATE *, size_t &);
  void simulateBlock(tHistory **, size_t *, size_t[][3]);
  void cleanGPU() {
    cudaFree(d_overflow);
    cudaFree(d_eTableMgr);
    cudaFree(d_oHisSize);
    cudaFree(d_oHisArr);
    cudaFree(d_oHisMax);
    cudaFree(d_gateMgr);
    for (auto &pair : d_wireMapper) {
      cudaFree(pair.second);
    }
  }

  void simulate(Gate *);
  void simulateBlock(vector<Gate *> &);

#ifdef GPU_DEBUG
  void checkEvalMgr(char *, size_t);
  void checkTrans(size_t, size_t);
  void checkGate(gpu_GATE *);
#endif
};

#endif
