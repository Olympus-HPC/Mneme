#include "Simulator.h"
#include "util.h"
#include <climits>
#include <chrono>
#include <thread>
using std::thread;

#include <stdio.h>

#ifdef GPU_DEBUG
#include <cmath>

void printEvalTable(char *h_eTable, size_t eSize) {
  char inLen;
  size_t tHeader;
  for (size_t i = 0; i < eSize; ++i) {
    tHeader = i * TRUTH_SIZE;
    inLen = h_eTable[tHeader + TRUTH_SIZE - 1];

    cout << (size_t)(h_eTable[tHeader + TRUTH_SIZE - 1]) << ' ';
    for (size_t j = 0; j < short(pow(4, inLen)); ++j)
      cout << (short)((h_eTable[tHeader + (j >> 2)] >> (2 * (j & 3))) & 3);
    cout << endl;
  }
}

void Simulator::checkEvalMgr(char *d_eTableMgr, size_t eSize) {
  char *h_eTableMgr = new char[TRUTH_SIZE * eSize];
  cudaMemcpy(h_eTableMgr, d_eTableMgr, sizeof(char) * TRUTH_SIZE * eSize,
             cudaMemcpyDeviceToHost);
  printEvalTable(h_eTableMgr, eSize);
  delete[] h_eTableMgr;
}

void printTransTable(Event *table, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    cout << table[i].t << ' ' << (short)table[i].v << endl;
  }
}

void Simulator::checkTrans(size_t wireSer, size_t size) {
  Event *d_tHis = d_wireMapper[wireSer];
  Event *h_tHis = new Event[size];
  cudaMemcpy(h_tHis, d_tHis, sizeof(Event) * size, cudaMemcpyDeviceToHost);
  cout << wireSer << ": " << endl;
  printTransTable(h_tHis, size);
  delete[] h_tHis;
}

void printGate(gpu_GATE *gate) {
  cout << "funcSer: " << gate->funcSer << endl;
  for (size_t i = 0; i < MAX_INPUT_PORT; ++i)
    cout << gate->iPort[i] << ' ';
  for (size_t i = 0; i < MAX_DTABLE; ++i)
    cout << gate->dTable[i] << ' ';
  cout << endl;
}

void Simulator::checkGate(gpu_GATE *d_gate) {
  gpu_GATE *h_gate = new gpu_GATE;
  cudaMemcpy(h_gate, d_gate, sizeof(gpu_GATE), cudaMemcpyDeviceToHost);
  printGate(h_gate);
  delete h_gate;
}

void printWireMapper(unordered_map<size_t, Event *> &WireMapper) {
  for (auto &d : WireMapper)
    cout << d.first << ' ' << d.second << endl;
}

#endif

Simulator::Simulator(tUnit off) : dumpOff(off) {
  cudaMalloc((void**)&d_overflow, sizeof(bool) * C_PARALLEL_LIMIT);
  cudaMalloc((void**)&d_oHisSize, sizeof(size_t) * C_PARALLEL_LIMIT);
  cudaMalloc((void**)&d_oHisMax, sizeof(size_t) * C_PARALLEL_LIMIT);
  cudaMalloc((void**)&d_oHisArr, sizeof(Event *) * C_PARALLEL_LIMIT);
  cudaMalloc((void**)&d_gateMgr, sizeof(gpu_GATE) * C_PARALLEL_LIMIT);
}

void Simulator::addEvalMgr(char *h_eTableMgr, size_t eSize) {
  cudaMalloc((void**)&d_eTableMgr, sizeof(char) * TRUTH_SIZE * eSize);
  cudaMemcpy(d_eTableMgr, h_eTableMgr, sizeof(char) * TRUTH_SIZE * eSize,
             cudaMemcpyHostToDevice);
#ifdef GPU_DEBUG
  // checkEvalMgr(d_eTableMgr, eSize);
#endif
}

void Simulator::addTrans(tHistory *trans, size_t &wireSer) {
  trans->push_end();
  cudaMalloc((void**)&d_HisTmp, sizeof(Event) * trans->size());
  cudaMemcpy(d_HisTmp, &(trans->front()), sizeof(Event) * trans->size(),
             cudaMemcpyHostToDevice);
  d_wireMapper.insert(make_pair(wireSer, d_HisTmp));
#ifdef GPU_DEBUG
  // checkTrans(wireSer, trans->size());
#endif
}

Event *Simulator::getTrans(size_t &wireSer) { return d_wireMapper[wireSer]; }

void Simulator::popTrans(size_t &wireSer) {
  cudaFree(d_wireMapper[wireSer]);
  d_wireMapper.erase(wireSer);
}

void Simulator::addGateMgr(gpu_GATE *h_gateMgr, size_t &simSize) {
  parallelSize = simSize;
  cudaMemcpy(d_gateMgr, h_gateMgr, sizeof(gpu_GATE) * simSize,
             cudaMemcpyHostToDevice);
#ifdef GPU_DEBUG
  printWireMapper(d_wireMapper);
  for (size_t i = 0; i < parallelSize; ++i) {
    checkGate(d_gateMgr + i);
  }
#endif
}

__global__ void simulateParallel(bool *overflow, const gpu_GATE *gateMgr,
                                 Event **oHisArr, const size_t *oHisMaxSize,
                                 size_t *oHisSize, const char *eTableMgr,
                                 size_t SimLimit, tUnit dumpOff) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid > SimLimit - 1)
    return;

  // Gate info
  size_t maxSize = oHisMaxSize[gid], currSize;
  const char *eTable = eTableMgr + gateMgr[gid].funcSer * TRUTH_SIZE;
  int8_t inLen = *(eTable + TRUTH_SIZE - 1);

  // Simulation cache
  Event *iPort[MAX_INPUT_PORT];
  Event *oPort = oHisArr[gid];

  tUnit currTime = -1, iTnext[MAX_INPUT_PORT] = {0};
  bool isInit = true;

  dUnit delay = -1, d;

  int16_t iVcurr = 0, iVprev = 0;

  int8_t oVcurr = 0, oVprev, i, iFrom, iTo, delayIdx;

  overflow[gid] = false;
  // Init first input pattern and first trans time
  for (i = 0; i < inLen; i++) {
    iPort[i] = gateMgr[gid].iPort[i];
    iVprev = iVprev << 2;
    iVprev |= iPort[i]->v;
    iTnext[i] = (iPort[i] + 1)->t;

    isInit = (iPort[i]->t > 0) ? false : isInit;
    currTime = (iTnext[i] < currTime) ? iTnext[i] : currTime;
  }

  // Init not simulated gate
  if (isInit) {
    currSize = 1;
    oPort[0] = {0, valueX};
    oVprev = valueX;
  } else {
    currSize = oHisSize[gid];
    oVprev = oPort[currSize - 1].v;
  }

  // Drop from full simulated gate
  if (currTime == (tUnit)-1 || currTime > dumpOff) {
    if (oPort[currSize - 1].t != (tUnit)-1) {
      oPort[currSize] = {(tUnit)-1, valueZ};
      oHisSize[gid] = currSize + 1;
    }
    return;
  }

  // Simulation loop
  while (currTime != (tUnit)-1 && currTime <= dumpOff) {
    // Update newly coming input pattern and corresponding output
    for (i = 0; i < inLen; ++i) {
      if (iTnext[i] == currTime) {
        ++iPort[i];
        iTnext[i] = (iPort[i] + 1)->t;
      }
      iVcurr = iVcurr << 2;
      iVcurr |= iPort[i]->v;
    }

    // Get new output value
    oVcurr = (eTable[iVcurr >> 2] >> ((iVcurr & 3) * 2)) & 3;

    // Trigger edge at output
    if (oVcurr != oVprev) {
      // Get minimum delay and next output trans time
      switch (oVprev << 2 | oVcurr) {
      case 1:
        delayIdx = 0;
        break; // 01
      case 4:
        delayIdx = 1;
        break; // 10
      case 2:
        delayIdx = 2;
        break; // 0X
      case 9:
        delayIdx = 3;
        break; // X1
      case 6:
        delayIdx = 4;
        break; // 1X
      case 8:
        delayIdx = 5;
        break; // X0
      }

      for (i = 0; i < inLen; ++i) {
        iTo = (iVcurr >> ((inLen - i - 1) << 1)) & 3,
        iFrom = (iVprev >> ((inLen - i - 1) << 1)) & 3;
        if (iTo != iFrom) {
          d = (iFrom == 1 || iTo == 0)
                  ? gateMgr[gid].dTable[MAX_DTUPLE * i + delayIdx + 6]
                  : gateMgr[gid].dTable[MAX_DTUPLE * i + delayIdx];

          delay = (d < delay) ? d : delay;
        }
      }

      currTime += delay;

      // Pop out late transtion
      while (currSize && oPort[currSize - 1].t >= currTime) {
        --currSize;
      }

      // Push back new output transtion
      if (!currSize || oPort[currSize - 1].v != oVcurr) {
        oPort[currSize] = {currTime, oVcurr};
        ++currSize;

        // Overflow
        if (currSize == maxSize) {
          overflow[gid] = true;
          oHisSize[gid] = currSize;
          return;
        }
      }
    }

    // Re-init cache
    iVprev = iVcurr;
    oVprev = oVcurr;
    iVcurr = 0;
    oVcurr = 0;
    currTime = -1;
    delay = -1;

    // Find next trans time
    for (i = 0; i < inLen; ++i) {
      currTime = (iTnext[i] < currTime) ? iTnext[i] : currTime;
    }
  }
  oPort[currSize] = {(tUnit)-1, valueZ};
  oHisSize[gid] = currSize + 1;
}

void Simulator::simulateBlock(tHistory **h_oWireArr, size_t *oWireSer,
                              size_t expArr[][3]) {
  bool finish;
  // Determine grid size
  size_t gridSize = (parallelSize + C_THREAD_LIMIT - 1) / C_THREAD_LIMIT;
#ifdef GPU_DEBUG
  cout << "Grid size " << gridSize << endl;
#endif
  // Allocate new output tHis mem in GPU
  for (size_t i = 0; i < parallelSize; ++i) {
    h_oHisMax[i] = expArr[i][0];
    cudaMalloc((void**)&d_HisTmp, sizeof(Event) * h_oHisMax[i]);
    h_oHisArr[i] = d_HisTmp;
  }

  auto start = std::chrono::steady_clock::now();

  size_t loop = 1;
  do {
    finish = true;
    // Copy output address
    cudaMemcpy(d_oHisArr, h_oHisArr, sizeof(Event *) * parallelSize,
               cudaMemcpyHostToDevice);

    // Copy Max Ouput size
    cudaMemcpy(d_oHisMax, h_oHisMax, sizeof(size_t) * parallelSize,
               cudaMemcpyHostToDevice);

    // Simulate
    simulateParallel<<<gridSize, C_THREAD_LIMIT>>>(
        d_overflow, d_gateMgr, d_oHisArr, d_oHisMax, d_oHisSize, d_eTableMgr,
        parallelSize, dumpOff);

    // Copy overflow flag
    cudaMemcpy(h_overflow, d_overflow, sizeof(bool) * parallelSize,
               cudaMemcpyDeviceToHost);

    // Handle overflow
    size_t n_oHisSize;
    for (size_t i = 0; i < parallelSize; ++i) {
      if (h_overflow[i]) {
        n_oHisSize = h_oHisMax[i] + expArr[i][loop % 3];

        cudaMalloc((void**)&d_HisTmp, sizeof(Event) * n_oHisSize);
        cudaMemcpy(d_HisTmp, h_oHisArr[i], sizeof(Event) * h_oHisMax[i],
                   cudaMemcpyDeviceToDevice);

        cudaFree(h_oHisArr[i]);
        h_oHisArr[i] = d_HisTmp;
        h_oHisMax[i] = n_oHisSize;
        finish = false;
      }
    }
    loop++;
#ifdef GPU_DEBUG
    cout << "Done loop " << loop << endl;
#endif
  } while (!finish);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total simulation time of %zu loops: %f (s)\n", loop, time * 1e-9f);

  // Copy back output tHis
  cudaMemcpy(h_oHisSize, d_oHisSize, sizeof(size_t) * parallelSize,
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < parallelSize; ++i) {
    h_oWireArr[i]->resize(h_oHisSize[i]);
    cudaMemcpy(h_oWireArr[i]->getHis(), h_oHisArr[i],
               sizeof(Event) * h_oHisSize[i], cudaMemcpyDeviceToHost);
    d_wireMapper.insert(make_pair(oWireSer[i], h_oHisArr[i]));
  }
}

void Simulator::simulate(Gate *g) {
  // Gate info
  vector<tHistory *> tList = g->getPortTable();
  const tTable *truthtable = g->getTruthTable();
  const dTable *delaytable = g->getDelayTable();
  size_t inLen = g->getInputSize(), delayIdx;

  // Simulation cache
  short iVcurr = 0, iVprev = 0;
  char oVcurr = tList[inLen]->getValue(0), oVprev = valueX, iFrom, iTo;
  size_t i;
  vector<size_t> iSize(inLen, 0), iHead(inLen, 0);
  vector<tUnit> iTnext(inLen, 0);
  tUnit currTime = -1, delay = -1, d;

  // Init input port trans len and first input pattern
  for (i = 0; i < inLen; i++) {
    iVprev = iVprev << 2;
    iVprev |= tList[i]->getValue(0);
    iSize[i] = tList[i]->size();
  }

  // Find first trans time
  for (i = 0; i < inLen; ++i) {
    if (1 < iSize[i]) {
      iTnext[i] = tList[i]->getTime(1);
      if (iTnext[i] < currTime)
        currTime = iTnext[i];
    }
  }

  // Simulation loop
  while (currTime != (tUnit)-1 && currTime <= dumpOff) {
    // Update newly coming input pattern and corresponding output
    for (i = 0; i < inLen; ++i) {
      if (iHead[i] + 1 < iSize[i]) {
        if (iTnext[i] == currTime)
          ++iHead[i];
      }

      iVcurr = iVcurr << 2;
      iVcurr |= tList[i]->getValue(iHead[i]);
    }

    oVcurr = truthtable->get(iVcurr);

    // Push back output port
    if (oVcurr != oVprev) {
      // Get minimum delay and next trans time
      switch (oVprev << 2 | oVcurr) {
      case 1:
        delayIdx = 0;
        break; // 01
      case 4:
        delayIdx = 1;
        break; // 10
      case 2:
        delayIdx = 2;
        break; // 0X
      case 9:
        delayIdx = 3;
        break; // X1
      case 6:
        delayIdx = 4;
        break; // 1X
      case 8:
        delayIdx = 5;
        break; // X0
      }

      for (i = 0; i < inLen; ++i) {
        iTo = (iVcurr >> ((inLen - i - 1) << 1)) & 3,
        iFrom = (iVprev >> ((inLen - i - 1) << 1)) & 3;
        if ((iTo) != (iFrom)) {
          d = (iFrom == 1 || iTo == 0)
                  ? (*(delaytable + (MAX_DTUPLE * i) + delayIdx + 6))
                  : // negedge
                  (*(delaytable + (MAX_DTUPLE * i) + delayIdx));

          if (d < delay)
            delay = d;
        }
      }
      currTime += delay;

      tList[inLen]->push_back(currTime, oVcurr);
    }

    // Update cache
    iVprev = iVcurr;
    oVprev = oVcurr;
    iVcurr = 0;
    oVcurr = 0;
    currTime = delay = -1;

    // Find next trans time
    for (i = 0; i < inLen; ++i) {
      if (iHead[i] + 1 < iSize[i]) {
        iTnext[i] = tList[i]->getTime(iHead[i] + 1);
        if (iTnext[i] < currTime)
          currTime = iTnext[i];
      }
    }
  }
}

void Simulator::simulateBlock(vector<Gate *> &gl) {
  vector<thread> threads;

  for (auto g : gl) {
    threads.push_back(thread(&Simulator::simulate, this, g));
  }

  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
