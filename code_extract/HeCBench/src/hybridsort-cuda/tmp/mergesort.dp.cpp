#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
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

SYCL_EXTERNAL
sycl::float4 sortElem(sycl::float4 r) {
  sycl::float4 nr;

  float xt = r.x();
  float yt = r.y();
  float zt = r.z();
  float wt = r.w();

  float nr_xt = xt > yt ? yt : xt;
  float nr_yt = yt > xt ? yt : xt;
  float nr_zt = zt > wt ? wt : zt;
  float nr_wt = wt > zt ? wt : zt;

  xt = nr_xt > nr_zt ? nr_zt : nr_xt;
  yt = nr_yt > nr_wt ? nr_wt : nr_yt;
  zt = nr_zt > nr_xt ? nr_zt : nr_xt;
  wt = nr_wt > nr_yt ? nr_wt : nr_yt;

  nr.x() = xt;
  nr.y() = yt > zt ? zt : yt;
  nr.z() = zt > yt ? zt : yt;
  nr.w() = wt;
  return nr;
}

void sortElement(sycl::float4 *result, sycl::float4 *input, const int size,
                 sycl::nd_item<3> item_ct1)
  {
  int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    if (gid < size) result[gid] = sortElem(input[gid]);
  }

SYCL_EXTERNAL
sycl::float4 getLowest(sycl::float4 a, sycl::float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  a.x() = ax < bw ? ax : bw;
  a.y() = ay < bz ? ay : bz;
  a.z() = az < by ? az : by;
  a.w() = aw < bx ? aw : bx;
  return a;
}

SYCL_EXTERNAL
sycl::float4 getHighest(sycl::float4 a, sycl::float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  b.x() = aw >= bx ? aw : bx;
  b.y() = az >= by ? az : by;
  b.z() = ay >= bz ? ay : bz;
  b.w() = ax >= bw ? ax : bw;
  return b;
}

// the kernel calls the functions defined above
#include "kernel_mergeSortPass.h"

void
mergepack ( float* result , 
const float* orig , 
const int *constStartAddr,
const unsigned int *finalStartAddr,
const unsigned int *nullElems ,
sycl::nd_item<3> item_ct1)
{

  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  int division = item_ct1.get_group(1);
          if((finalStartAddr[division] + gid) < finalStartAddr[division + 1])
          result[finalStartAddr[division] + gid] = 
          orig[constStartAddr[division]*4 + nullElems[division] + gid];
}

sycl::float4 *runMergeSort(int listsize, int divisions,
                           sycl::float4 *d_origList, sycl::float4 *d_resultList,
                           int *sizes, int *nullElements,
                           unsigned int *origOffsets) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

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
  sycl::float4 *d_resultList_buff;
  d_resultList_buff = (sycl::float4 *)sycl::malloc_device(
      sizeof(sycl::float4) * listsize / 4, q_ct1);

  sycl::float4 *d_origList_buff;
  d_origList_buff = (sycl::float4 *)sycl::malloc_device(
      sizeof(sycl::float4) * listsize / 4, q_ct1);
  q_ct1.memcpy(d_origList_buff, d_origList,
               sizeof(sycl::float4) * listsize / 4);

  int* d_constStartAddr;
  d_constStartAddr = sycl::malloc_device<int>((divisions + 1), q_ct1);
  q_ct1.memcpy(d_constStartAddr, startaddr, sizeof(int) * (divisions + 1));

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                           sycl::range<3>(1, 1, THREADS),
                                       sycl::range<3>(1, 1, THREADS)),
                     [=](sycl::nd_item<3> item_ct1) {
                       sortElement(d_resultList_buff, d_origList_buff,
                                   listsize / 4, item_ct1);
                     });
  });

  //double mergePassTime = 0;
  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    int threadsPerDiv = (int)ceil(largestSize / (float)floatsperthread);
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
    sycl::float4 *tempList = d_origList_buff;
    d_origList_buff = d_resultList_buff;
    d_resultList_buff = tempList;

    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid[0]) *
                                             sycl::range<3>(1, 1, local[0]),
                                         sycl::range<3>(1, 1, local[0])),
                       [=](sycl::nd_item<3> item_ct1) {
                         mergeSortPass(d_origList_buff, d_resultList_buff,
                                       d_constStartAddr, threadsPerDiv, nrElems,
                                       item_ct1);
                       });
    });

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
  //global[0] = grid[0]*local[0];
  //global[1] = grid[1]*local[1];

  unsigned int *d_finalStartAddr;
  d_finalStartAddr = sycl::malloc_device<unsigned int>((divisions + 1), q_ct1);
  q_ct1.memcpy(d_finalStartAddr, origOffsets,
               sizeof(unsigned int) * (divisions + 1));

  unsigned int *d_nullElements;
  d_nullElements = sycl::malloc_device<unsigned int>(divisions, q_ct1);
  q_ct1.memcpy(d_nullElements, nullElements, sizeof(unsigned int) * divisions);

  sycl::range<3> grids(grid[0], grid[1], 1);
  sycl::range<3> threads(local[0], local[1], 1);

  q_ct1.submit([&](sycl::handler &cgh) {
    auto dpct_global_range = grids * threads;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          mergepack((float *)d_origList_buff, (float *)d_resultList_buff,
                    d_constStartAddr, d_finalStartAddr, d_nullElements,
                    item_ct1);
        });
  });

  q_ct1.memcpy(d_origList, d_origList_buff, sizeof(sycl::float4) * listsize / 4)
      .wait();

  sycl::free(d_resultList_buff, q_ct1);
  sycl::free(d_origList_buff, q_ct1);
  sycl::free(d_finalStartAddr, q_ct1);
  sycl::free(d_constStartAddr, q_ct1);
  sycl::free(d_nullElements, q_ct1);
  free(startaddr);
  return d_origList;
}
