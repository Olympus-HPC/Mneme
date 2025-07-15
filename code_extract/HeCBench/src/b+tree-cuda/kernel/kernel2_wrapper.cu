#include <cuda.h>
#include <stdio.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel2.cu"
#include "./kernel2_wrapper.h"

void 
kernel2_wrapper(  knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    long *lastKnode,
    long *offset_2,
    int *start,
    int *end,
    int *recstart,
    int *reclength)
{

  long long offload_start = get_time();

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);


  //==================================================50
  //  knodesD
  //==================================================50

  knode *knodesD;
  cudaMalloc((void**)&knodesD, knodes_mem);

  //==================================================50
  //  currKnodeD
  //==================================================50

  long *currKnodeD;
  cudaMalloc((void**)&currKnodeD, count*sizeof(long));

  //==================================================50
  //  offsetD
  //==================================================50

  long *offsetD;
  cudaMalloc((void**)&offsetD, count*sizeof(long));

  //==================================================50
  //  lastKnodeD
  //==================================================50

  long *lastKnodeD;
  cudaMalloc((void**)&lastKnodeD, count*sizeof(long));

  //==================================================50
  //  offset_2D
  //==================================================50

  long *offset_2D;
  cudaMalloc((void**)&offset_2D, count*sizeof(long));

  //==================================================50
  //  startD
  //==================================================50

  int *startD;
  cudaMalloc((void**)&startD, count*sizeof(int));

  //==================================================50
  //  endD
  //==================================================50

  int *endD;
  cudaMalloc((void**)&endD, count*sizeof(int));

  //==================================================50
  //  ansDStart
  //==================================================50

  int *ansDStart;
  cudaMalloc((void**)&ansDStart, count*sizeof(int));

  //==================================================50
  //  ansDLength
  //==================================================50

  int *ansDLength;
  cudaMalloc((void**)&ansDLength, count*sizeof(int));

  cudaMemcpyAsync(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(currKnodeD, currKnode, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(offsetD, offset, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(lastKnodeD, lastKnode, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(offset_2D, offset_2, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(startD, start, count*sizeof(int), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(endD, end, count*sizeof(int), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(ansDStart, recstart, count*sizeof(int), cudaMemcpyHostToDevice, 0);

  cudaMemcpyAsync(ansDLength, reclength, count*sizeof(int), cudaMemcpyHostToDevice, 0);

  cudaDeviceSynchronize();
  long long kernel_start = get_time();
   
  // [GPU] findRangeK kernel
  findRangeK<<<numBlocks, threadsPerBlock>>>(  maxheight,
      knodesD,
      knodes_elem,
      currKnodeD,
      offsetD,
      lastKnodeD,
      offset_2D,
      startD,
      endD,
      ansDStart,
      ansDLength);

  cudaDeviceSynchronize();
  long long kernel_end = get_time();
  printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));

  cudaMemcpyAsync(recstart, ansDStart, count*sizeof(int), cudaMemcpyDeviceToHost, 0);

  cudaMemcpyAsync(reclength, ansDLength, count*sizeof(int), cudaMemcpyDeviceToHost, 0);

  cudaFree(knodesD);
  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(lastKnodeD);
  cudaFree(offset_2D);
  cudaFree(startD);
  cudaFree(endD);
  cudaFree(ansDStart);
  cudaFree(ansDLength);

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif
}

