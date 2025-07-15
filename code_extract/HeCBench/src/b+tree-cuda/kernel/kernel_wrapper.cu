#include <cuda.h>
#include <stdio.h>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel.cu"
#include "./kernel_wrapper.h"        // (in current directory)

void 
kernel_wrapper(record *records,
    long records_mem,
    knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{

  int numBlocks;
  numBlocks = count;                  // max # of blocks can be 65,535
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

  //==================================================50
  //  recordsD
  //==================================================50

  record *recordsD;
  cudaMalloc((void**)&recordsD, records_mem);

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
  //  keysD
  //==================================================50

  int *keysD;
  cudaMalloc((void**)&keysD, count*sizeof(int));

  //==================================================50
  //  ansD
  //==================================================50

  record *ansD;
  cudaMalloc((void**)&ansD, count*sizeof(record));

  //==================================================50
  //  recordsD
  //==================================================50

  cudaMemcpyAsync(recordsD, records, records_mem, cudaMemcpyHostToDevice, 0);

  //==================================================50
  //  knodesD
  //==================================================50

  cudaMemcpyAsync(knodesD, knodes, knodes_mem, cudaMemcpyHostToDevice, 0);

  //==================================================50
  //  currKnodeD
  //==================================================50

  cudaMemcpyAsync(currKnodeD, currKnode, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  //==================================================50
  //  offsetD
  //==================================================50

  cudaMemcpyAsync(offsetD, offset, count*sizeof(long), cudaMemcpyHostToDevice, 0);

  //==================================================50
  //  keysD
  //==================================================50

  cudaMemcpyAsync(keysD, keys, count*sizeof(int), cudaMemcpyHostToDevice, 0);

  //==================================================50
  //  ansD
  //==================================================50

  cudaMemcpyAsync(ansD, ans, count*sizeof(record), cudaMemcpyHostToDevice, 0);

  cudaDeviceSynchronize();
  long long kernel_start = get_time();

  findK<<<numBlocks, threadsPerBlock>>>(  maxheight,
      knodesD,
      knodes_elem,
      recordsD,
      currKnodeD,
      offsetD,
      keysD,
      ansD);

  cudaDeviceSynchronize();
  long long kernel_end = get_time();
  printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));

  cudaMemcpy(ans, ansD, count*sizeof(record), cudaMemcpyDeviceToHost);

  cudaFree(recordsD);
  cudaFree(knodesD);
  cudaFree(currKnodeD);
  cudaFree(offsetD);
  cudaFree(keysD);
  cudaFree(ansD);

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif
}
