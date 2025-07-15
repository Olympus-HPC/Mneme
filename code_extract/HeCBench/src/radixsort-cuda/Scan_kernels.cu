/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

static unsigned int iSnapUp(const unsigned int dividend, const unsigned int divisor)
{
  return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
}
unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
{
  if(!L)
  {
    log2L = 0;
    return 0;
  } else {
    for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
    return L;
  }
}


////////////////////////////////////////////////////////////////////////////////
// Scan codelets
////////////////////////////////////////////////////////////////////////////////
#if(1)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
__device__
inline unsigned int scan1Inclusive(const unsigned int idata, 
                           unsigned int* l_Data, const unsigned int size)
{
  unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;

  for(unsigned int offset = 1; offset < size; offset <<= 1){
    __syncthreads();
    unsigned int t = l_Data[pos] + l_Data[pos - offset];
    __syncthreads();
    l_Data[pos] = t;
  }

  return l_Data[pos];
}

__device__
inline unsigned int scan1Exclusive(const unsigned int idata, 
                                   unsigned int* l_Data, const unsigned int size)
{
  return scan1Inclusive(idata, l_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define      WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naiveScan1 but doesn't need barriers
//assuming size <= WARP_SIZE
inline unsigned int warpScanInclusive(const unsigned int idata, 
                                      unsigned int* l_Data, const unsigned int size)
{
  unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
  l_Data[pos] = 0;
  pos += size;
  l_Data[pos] = idata;

  for(unsigned int offset = 1; offset < size; offset <<= 1)
    l_Data[pos] += l_Data[pos - offset];

  return l_Data[pos];
}

inline unsigned int warpScanExclusive(const unsigned int idata, 
                                      unsigned int* l_Data, const unsigned int size)
{
  return warpScanInclusive(idata, l_Data, size) - idata;
}

__device__
inline unsigned int scan1Inclusive(const unsigned int idata, 
                                   unsigned int* l_Data, const unsigned int size)
{
  if(size > WARP_SIZE){
    //Bottom-level inclusive warp scan
    unsigned int warpResult = warpScanInclusive(idata, l_Data, WARP_SIZE);

    //Save top elements of each warp for exclusive warp scan
    //sync to wait for warp scans to complete (because l_Data is being overwritten)
    __syncthreads();

    int lid = threadIdx.x;
    if( (lid & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
      l_Data[lid >> LOG2_WARP_SIZE] = warpResult;

    //wait for warp scans to complete
    __syncthreads();
    if( lid < (WORKGROUP_SIZE / WARP_SIZE) ){
      //grab top warp elements
      unsigned int val = l_Data[lid] ;
      //calculate exclsive scan and write back to shared memory
      l_Data[lid] = warpScanExclusive(val, l_Data, size >> LOG2_WARP_SIZE);
    }

    //return updated warp scans with exclusive scan results
    __syncthreads();
    return warpResult + l_Data[lid >> LOG2_WARP_SIZE];
  }else{
    return warpScanInclusive(idata, l_Data, size);
  }
}

__device__
inline unsigned int scan1Exclusive(const unsigned int idata, 
                                   unsigned int* l_Data, const unsigned int size){
  return scan1Inclusive(idata, l_Data, size) - idata;
}
#endif


//Vector scan: the array to be scanned is stored
//in work-item private memory as uint4
__device__
inline uint4 scan4Inclusive(uint4 data4, 
                            unsigned int* l_Data, const unsigned int size){
  //Level-0 inclusive scan
  data4.y += data4.x;
  data4.z += data4.y;
  data4.w += data4.z;

  //Level-1 exclusive scan
  unsigned int val = scan1Inclusive(data4.w, l_Data, size / 4) - data4.w;

  return (data4 + make_uint4(val));
}

__device__
inline uint4 scan4Exclusive(uint4 data4, 
                            unsigned int* l_Data, const unsigned int size)
{
  return scan4Inclusive(data4, l_Data, size) - data4;
}


////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__  void scanExclusiveLocal1K(
            unsigned int*__restrict__ d_Dst,
      const unsigned int*__restrict__ d_Src,
      const unsigned int size)
{
    __shared__ unsigned int l_Data[2 * WORKGROUP_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = reinterpret_cast<const uint4*>(d_Src)[i];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, l_Data, size);

    //Write back
    reinterpret_cast<uint4*>(d_Dst)[i] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveLocal2K(
            unsigned int*__restrict__ d_Buf,
            unsigned int*__restrict__ d_Dst,
      const unsigned int*__restrict__ d_Src,
      const unsigned int N,
      const unsigned int arrayLength)
{
    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    //Skip loads and stores for inactive work-items of the work-group with highest index(pos >= N)

    __shared__ unsigned int l_Data[2 * WORKGROUP_SIZE];
    unsigned int data = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
      data = d_Dst[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i] + 
             d_Src[(4 * WORKGROUP_SIZE - 1) + (4 * WORKGROUP_SIZE) * i];

    //Compute
    unsigned int odata = scan1Exclusive(data, l_Data, arrayLength);

    //Avoid out-of-bound access
    if(i < N) d_Buf[i] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdateK(
      unsigned int*__restrict__ d_Data,
      unsigned int*__restrict__ d_Buf)
{
    __shared__ unsigned int buf[1];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint4 data4 = reinterpret_cast<uint4*>(d_Data)[i];
    if(threadIdx.x == 0)
      buf[0] = d_Buf[blockIdx.x];

    __syncthreads();
    data4 += make_uint4(buf[0]);

    reinterpret_cast<uint4*>(d_Data)[i] = data4;
}
