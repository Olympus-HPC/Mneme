#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include "mergesort.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE  256
#define ROW_LENGTH  BLOCKSIZE * 4
#define ROWS    4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////


__device__
float4 sortElem(float4 r) {
  float4 nr;

  float xt = r.x;
  float yt = r.y;
  float zt = r.z;
  float wt = r.w;

  float nr_xt = xt > yt ? yt : xt;
  float nr_yt = yt > xt ? yt : xt;
  float nr_zt = zt > wt ? wt : zt;
  float nr_wt = wt > zt ? wt : zt;

  xt = nr_xt > nr_zt ? nr_zt : nr_xt;
  yt = nr_yt > nr_wt ? nr_wt : nr_yt;
  zt = nr_zt > nr_xt ? nr_zt : nr_xt;
  wt = nr_wt > nr_yt ? nr_wt : nr_yt;

  nr.x = xt;
  nr.y = yt > zt ? zt : yt;
  nr.z = zt > yt ? zt : yt;
  nr.w = wt;
  return nr;
}

  __global__ void
sortElement(float4* result, float4* input, const int size) 
{
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  if (gid < size) result[gid] = sortElem(input[gid]);
}

  __device__
float4 getLowest(float4 a, float4 b)
{
  float ax = a.x;
  float ay = a.y;
  float az = a.z;
  float aw = a.w;
  float bx = b.x;
  float by = b.y;
  float bz = b.z;
  float bw = b.w;
  a.x = ax < bw ? ax : bw;
  a.y = ay < bz ? ay : bz;
  a.z = az < by ? az : by;
  a.w = aw < bx ? aw : bx;
  return a;
}

  __device__
float4 getHighest(float4 a, float4 b)
{
  float ax = a.x;
  float ay = a.y;
  float az = a.z;
  float aw = a.w;
  float bx = b.x;
  float by = b.y;
  float bz = b.z;
  float bw = b.w;
  b.x = aw >= bx ? aw : bx;
  b.y = az >= by ? az : by;
  b.z = ay >= bz ? ay : bz;
  b.w = ax >= bw ? ax : bw;
  return b;
}

// the kernel calls the functions defined above
#include "kernel_mergeSortPass.h"

  __global__ void
mergepack ( float* result , 
    const float* orig , 
    const int *constStartAddr,
    const unsigned int *finalStartAddr,
    const unsigned int *nullElems )
{

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int division = blockIdx.y;
  if((finalStartAddr[division] + gid) < finalStartAddr[division + 1])
    result[finalStartAddr[division] + gid] = 
      orig[constStartAddr[division]*4 + nullElems[division] + gid];
}


float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets){

  int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
  int largestSize = -1;
  startaddr[0] = 0;
  for(int i=1; i<=divisions; i++)
  {
    startaddr[i] = startaddr[i-1] + sizes[i-1];
    if(sizes[i-1] > largestSize) largestSize = sizes[i-1];
  }
  largestSize *= 4;

#ifdef MERGE_WG_SIZE_0
  const int THREADS = MERGE_WG_SIZE_0;
#else
  const int THREADS = 256;
#endif
  size_t local[] = {THREADS,1,1};
  size_t blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
  size_t grid[3];

  // divided by four 
  float4 *d_resultList_buff;
  cudaMalloc((void**)&d_resultList_buff, sizeof(float4)*listsize/4);

  float4 *d_origList_buff;
  cudaMalloc((void**)&d_origList_buff, sizeof(float4)*listsize/4);
  cudaMemcpyAsync(d_origList_buff, d_origList, sizeof(float4)*listsize/4, cudaMemcpyHostToDevice, 0);

  int* d_constStartAddr;
  cudaMalloc((void**)&d_constStartAddr, sizeof(int)*(divisions+1));
  cudaMemcpyAsync(d_constStartAddr, startaddr, sizeof(int)*(divisions+1), cudaMemcpyHostToDevice, 0);

  sortElement<<<blocks, THREADS>>>(d_resultList_buff, d_origList_buff, listsize/4);

  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread);
    int threadsNeeded = threadsPerDiv * divisions;

#ifdef MERGE_WG_SIZE_1
    local[0] = MERGE_WG_SIZE_1;
#else
    local[0] = 208;
#endif

    grid[0] = ((threadsNeeded%local[0]) == 0) ?
      threadsNeeded/local[0] :
      (threadsNeeded/local[0]) + 1;
    if(grid[0] < 8){
      grid[0] = 8;
      local[0] = ((threadsNeeded%grid[0]) == 0) ?
        threadsNeeded / grid[0] :
        (threadsNeeded / grid[0]) + 1;
    }
    // Swap orig/result list
    float4* tempList = d_origList_buff;
    d_origList_buff = d_resultList_buff;
    d_resultList_buff = tempList;

    mergeSortPass<<<grid[0], local[0]>>>(d_origList_buff, d_resultList_buff, 
        d_constStartAddr, threadsPerDiv, nrElems, listsize/4);

    nrElems *= 2;
    floatsperthread = (nrElems*4);

    if(threadsPerDiv == 1) break;
  }


#ifdef MERGE_WG_SIZE_0
  local[0] = MERGE_WG_SIZE_0;
#else
  local[0] = 256;
#endif
  grid[0] = ((largestSize%local[0]) == 0) ?  largestSize/local[0] : (largestSize/local[0]) + 1;
  grid[1] = divisions;

  unsigned int *d_finalStartAddr;
  cudaMalloc((void**)&d_finalStartAddr, sizeof(unsigned int)*(divisions+1));
  cudaMemcpyAsync(d_finalStartAddr, origOffsets,  sizeof(unsigned int)*(divisions+1), cudaMemcpyHostToDevice,0);

  unsigned int *d_nullElements;
  cudaMalloc((void**)&d_nullElements, sizeof(unsigned int)*divisions);
  cudaMemcpyAsync(d_nullElements, nullElements, sizeof(unsigned int)*divisions, cudaMemcpyHostToDevice,0);

  dim3 grids(grid[0], grid[1]); 
  dim3 threads(local[0], local[1]); 

  mergepack<<<grids, threads>>>(
      (float*)d_origList_buff, 
      (float*)d_resultList_buff, 
      d_constStartAddr,
      d_finalStartAddr, 
      d_nullElements);

  cudaMemcpy(d_origList, d_origList_buff, sizeof(float4)*listsize/4, cudaMemcpyDeviceToHost);

  cudaFree(d_resultList_buff);
  cudaFree(d_origList_buff);
  cudaFree(d_finalStartAddr);
  cudaFree(d_constStartAddr);
  cudaFree(d_nullElements);
  free(startaddr);
  return d_origList;
}
