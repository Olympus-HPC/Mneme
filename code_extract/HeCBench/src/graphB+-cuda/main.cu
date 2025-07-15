/*
graphB+ balancing algorithm for signed social network graphs

Copyright 2021, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Ghadeer Alabandi and Martin Burtscher

URL: The latest version of this code is available at
https://cs.txstate.edu/~burtscher/research/graphB/.
*/


#include <cstdio>
#include <chrono>
#include <climits>
#include <algorithm>
#include <set>
#include <map>
#include <cuda.h>
#include "kernels.h"

int main(int argc, char* argv[])
{
  printf("graphB+ balancing code for signed social network graphs (%s)\n", __FILE__);
  printf("Copyright 2021 Texas State University\n");
  cudaSetDevice(Device);

  // process command line and read input
  if (argc != 4) {printf("USAGE: %s input_file_name iteration_count output_file_name\n", argv[0]); exit(-1);}

#ifdef VERIFY
  printf("verification is on\n");
#else
  printf("verification is off\n");
#endif
  printf("input: %s\n", argv[1]);
  Graph g = readGraph(argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int iterations = atoi(argv[2]);

  // allocate all memory
  bool* const minus = new bool [g.edges];
  int* const parent = new int [g.nodes];
  int* const queue = new int [g.nodes];  // first used as queue, then as CC size
  int* const label = new int [g.nodes];  // first used as count, then as label, and finally as CC label
  int* const border = new int [g.nodes + 2];  // maybe make smaller
  int* const inCC = new int [g.nodes];  // how often node was in largest CC or at an even distance from largest CC
  int* const inTree = new int [g.edges];  // how often edge was in tree
  int* const negCnt = new int [g.edges];  // how often edge was negative
  EdgeInfo* const einfo = new EdgeInfo [g.edges + 16];
  int* const root = new int [g.nodes];  // tree roots

  for (int i = 0; i < g.nodes; i++) root[i] = i;
  std::partial_sort(root, root + std::min(iterations, g.nodes), root + g.nodes, [&](int a, int b) {
    return (g.nindex[a + 1] - g.nindex[a]) > (g.nindex[b + 1] - g.nindex[b]);
  });

  //GPU code
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("Device: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
         deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  Graph d_g = g;
  EdgeInfo* d_einfo;
  int* d_label;
  int* d_parent;
  int* d_queue;
  int* d_border;
  int* d_tail;
  int* d_inCC;
  int* d_inTree;
  int* d_negCnt;
  int* d_ws1;
  int* d_ws2;
  int* d_wSize;
  bool* d_minus;
  bool* changed_gpu;

  if (cudaSuccess != cudaMalloc((void **)&d_g.eweight, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");
 
  if (cudaSuccess != cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_inTree, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_negCnt, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_inCC, sizeof(int) * g.nodes))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_einfo, sizeof(EdgeInfo) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_label, sizeof(int) * g.nodes))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_parent, sizeof(int) * g.nodes))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_border, sizeof(int) * (g.nodes + 2)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_queue, sizeof(int) * g.nodes))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_tail, sizeof(int)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_ws1, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_ws2, sizeof(int) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_wSize, sizeof(int)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&d_minus, sizeof(bool) * g.edges))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMalloc((void **)&changed_gpu, sizeof(bool)))
    fprintf(stderr, "ERROR: could not allocate memory\n");

  if (cudaSuccess != cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice))
    fprintf(stderr, "ERROR: copying to device failed\n");

  if (cudaSuccess != cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice))
    fprintf(stderr, "ERROR: copying to device failed\n");

  if (cudaSuccess != cudaMemcpy(d_g.eweight, g.eweight, sizeof(int) * g.edges, cudaMemcpyHostToDevice))
    fprintf(stderr, "ERROR: copying to device failed\n");

  const int blocks = SMs * mTpSM / ThreadsPerBlock;

  // use random pluses and minuses
  init<<<blocks, ThreadsPerBlock>>>(g.edges, g.nodes, d_g.nlist, d_g.eweight, d_inCC, d_einfo, d_inTree, d_negCnt);

  cudaDeviceSynchronize();

  int min_d = INT_MAX;
  int max_d = INT_MIN;
  int sum_d = 0;
  double avg_d = 0;

  auto start = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < iterations; iter++) {
    // generate tree
    init2<<<blocks, ThreadsPerBlock>>>(g.edges, g.nodes, root[iter % g.nodes], d_g.nlist, d_parent, d_queue, d_label, d_tail);
    int level = 0;
    int tail = 1;
    border[0] = 0;
    border[1] = tail;
    while (border[level + 1] < g.nodes) {
      generateSpanningTree<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, iter + 17, d_einfo, d_parent, d_queue, level, d_tail, border[level],  border[level + 1]);
      if (cudaSuccess != cudaMemcpy(&tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost)) 
        fprintf(stderr, "ERROR: copying to host failed \n");
      level++;
      border[level + 1] = tail;
    }

    const int levels = level + 1;

    //min , max and avg depth of the trees
    sum_d += level;
    if (level < min_d) min_d = level;
    if (level > max_d) max_d = level;

#ifdef VERIFY
    verify_generateSpanningTree<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges, d_g.nindex, d_g.nlist, iter, d_parent,
                                                             level, d_tail, border[level + 1]);
#endif
    //root count
    //#1

    for (int level = levels - 1; level > 0; level--) {
      rootcount<<<blocks, ThreadsPerBlock>>>(d_parent, d_queue, d_label, level, border[level],  border[level + 1]);
    }

#ifdef VERIFY
    if (cudaSuccess != cudaMemcpy((void *)&label[root[iter % g.nodes]], (void *)&d_label[root[iter % g.nodes]],
                                  sizeof(int), cudaMemcpyDeviceToHost))
    fprintf(stderr, "ERROR: copying to host failed\n");

    if (label[root[iter % g.nodes]] != g.nodes)
    printf("ERROR: root count mismatch\n");
#endif

    // tree label
    label[root[iter % g.nodes]] = 0;
    if (cudaSuccess != cudaMemset((void*)&d_label[root[iter % g.nodes]], 0, sizeof(int)))
      fprintf(stderr, "ERROR: setting d_label failed\n");

    //#2
    for (int level = 0; level < levels; level++) {
      treelabel<<<blocks, ThreadsPerBlock>>>(
        g.nodes, d_g.nindex, d_g.nlist, d_einfo, d_inTree, d_negCnt, d_parent, 
        d_queue, d_label, level, border[level], border[level + 1]);
    }

    //#3
    inTreeUpdate<<<blocks, ThreadsPerBlock>>>(g.edges, d_g.nlist, d_inTree);
    initMinus<<<blocks, ThreadsPerBlock>>>(g.edges, g.nodes, d_g.nindex, d_g.nlist, d_einfo, d_minus);

    //#4
    processCycles<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, d_label, d_einfo, d_minus);

    init3<<<blocks, ThreadsPerBlock>>> (g.nodes, d_g.nindex, d_g.nlist, d_label, d_queue);
    compute1<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, d_label, d_minus, d_negCnt);
    flatten<<<blocks, ThreadsPerBlock>>>(g.nodes, d_label);
    ccSize<<<blocks, ThreadsPerBlock>>>(g.nodes, d_label, d_queue);
    largestCC<<<blocks, ThreadsPerBlock>>>(g.nodes, d_queue);
    ccHopCount<<<blocks, ThreadsPerBlock>>>(g.nodes, d_g.nindex, d_g.nlist, d_label, d_queue, d_ws1, d_ws2);

    bool changed;
    do {
      changed = false;
      if (cudaSuccess != cudaMemset(changed_gpu, 0, sizeof(bool)))
        fprintf(stderr, "ERROR: setting changed failed\n");

      BellmanFord<<<blocks, ThreadsPerBlock>>>(d_queue, changed_gpu, d_ws1, d_ws2);
      BellmanFord<<<blocks, ThreadsPerBlock>>>(d_queue, changed_gpu, d_ws1, d_ws2);
      BellmanFord<<<blocks, ThreadsPerBlock>>>(d_queue, changed_gpu, d_ws1, d_ws2);

      if (cudaSuccess != cudaMemcpy(&changed, changed_gpu, sizeof(bool), cudaMemcpyDeviceToHost))
        fprintf(stderr, "ERROR: copying of changed from device failed\n");
    } while (changed);

    incrementCC<<<blocks, ThreadsPerBlock>>>(g.nodes, d_label, d_queue, d_inCC);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = elapsed_seconds.count();
  printf("Total graphB+ runtime:    %.6f s\n", runtime);

  if (cudaSuccess != cudaMemcpy(inCC, d_inCC, sizeof(int) * g.nodes, cudaMemcpyDeviceToHost))
  fprintf(stderr, "ERROR: copying incc from device failed\n");

  // print results
  avg_d = sum_d / iterations;
  printf("number of trees %d\n", iterations);
  printf("Min depth of the trees %d\n Max depth of the trees %d\n Avg depth of the trees %.4f\n", min_d, max_d, avg_d);
  for (int i = 0; i < g.nodes; i++)
  {
    if (i >= 10)
    break; // to limit output
    printf("%6d: %6d   (%5.1f%%)  %d\n", i, inCC[i], 100.0 * inCC[i] / iterations, g.origID[i]);
  }

  // output results to file
  FILE *f = fopen(argv[3], "wt");
  fprintf(f, "original node ID, percentage node was in agreeable majority\n");
  for (int i = 0; i < g.nodes; i++) {
    fprintf(f, "%d,%.1f\n", g.origID[i], 100.0 * inCC[i] / iterations);
  }
  fprintf(f, "source node ID, destination node ID, percentage edge was in tree, percentage edge was negative\n");
  for (int v = 0; v < g.nodes; v++) {
    for (int j = g.nindex[v]; j < g.nindex[v + 1]; j++) {
      const int n = g.nlist[j] >> 1;
      if (v < n) {  // only print one copy of each edge (other copy does not have correct negCnt)
        fprintf(f, "%d,%d,%.1f,%.1f\n", g.origID[v], g.origID[n], 100.0 * inTree[j] / iterations, 100.0 * negCnt[j] / iterations);
      }
    }
  }
  fclose(f);

  // finalize
  freeGraph(g);
  delete [] minus;
  delete [] einfo;
  delete [] parent;
  delete [] queue;
  delete [] label;
  delete [] border;
  delete [] inCC;
  delete [] inTree;
  delete [] negCnt;
  delete [] root;
  cudaFree(d_g.nlist);
  cudaFree(d_g.nindex);
  cudaFree(d_einfo);
  cudaFree(d_inCC);
  cudaFree(d_negCnt);
  cudaFree(d_inTree);
  cudaFree(d_label);
  cudaFree(d_parent);
  cudaFree(d_queue);
  cudaFree(d_border);
  cudaFree(d_tail);
  cudaFree(changed_gpu);
  cudaFree(d_ws1);
  cudaFree(d_ws2);
  cudaFree(d_wSize);
  cudaFree(d_minus);
  // cudaFree(d_ws1);

  return 0;
}
