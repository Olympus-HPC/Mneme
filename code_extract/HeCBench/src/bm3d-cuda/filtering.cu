#include <float.h>
#include <stdio.h>
#include <cuda.h>

#include "indices.hpp"
#include "params.hpp"


// Kernels used for collaborative filtering and aggregation

//Sum the passed values in a warp to the first thread of this warp.
template<typename T>
__device__ inline T warpReduceSum(T val) 
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val,offset);
  return val;
}


//Sum the passed values in a block to the first thread of a block.
template<typename T>
__inline__ __device__ float blockReduceSum(T* shared, T val, int tid, int tcount) 
{
  int lane = tid % warpSize;
  int wid = tid / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (tid < tcount / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

//Returns absolute value of the passed real number raised to the power of two
__device__ __forceinline__
float abspow2(float & a)
{
  return a * a;
}


//Integer logarithm base 2.
template <typename IntType>
__device__ __inline__ uint ilog2(IntType n)
{
  uint l;
  for (l = 0; n; n >>= 1, ++l);
  return l;
}


//Orthogonal transformation.
template <typename T>
__device__ __inline__ void rotate(T& a, T& b)
{
  T tmp;
  tmp = a;
  a = tmp + b;
  b = tmp - b;
}


//Fast Walsh-Hadamard transform.
template <typename T>
__device__ __inline__ void fwht(T *data, uint n)
{
  unsigned l2 = ilog2(n) - 1;
  for ( uint i = 0; i < l2; ++i )
  {
    for (uint j = 0; j < n; j += (1 << (i + 1)))
    for (uint k = 0; k < (uint)(1 << i); ++k)
      rotate(data[j + k], data[j + k + (uint)(1 << i)]);
  }
}

//Based on blockIdx it computes the addresses to the arrays in global memory
__device__ inline void get_block_addresses(
  const uint2 & start_point,    //IN: first reference patch of a batch
  const uint & patch_stack_size,  //IN: maximal size of a 3D group
  const uint2 & stacks_dim,    //IN: Size of area, where reference patches could be located
  const Params & params,      //IN: Denoising parameters
  uint2 & outer_address,      //OUT: Coordinetes of reference patch in the image
  uint & start_idx)        //OUT: Address of a first element of the 3D group in stacks array
{
  //One block handles one patch_stack, data are in array one after one.
  start_idx = patch_stack_size * idx2(blockIdx.x,blockIdx.y,gridDim.x);
  
  outer_address.x = start_point.x + (blockIdx.x * params.p);
  outer_address.y = start_point.y + (blockIdx.y * params.p);

  //Ensure, that the bottom most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.y >= stacks_dim.y && outer_address.y < stacks_dim.y + params.p - 1)
    outer_address.y = stacks_dim.y - 1;
  //Ensure, that the right most patches will be taken as reference patches regardless the p parameter.
  if (outer_address.x >= stacks_dim.x && outer_address.x < stacks_dim.x + params.p - 1)
    outer_address.x = stacks_dim.x - 1;
}

/*
Gather patches form image based on matching stored in 3D array stacks
Used parameters: p,k,N
Division: One block handles one patch_stack, threads match to the pixels of a patch
*/
__global__
void get_block(
    const uint2 start_point,                       //IN: first reference patch of a batch
    const uchar* __restrict image,                 //IN: image
    const ushort* __restrict stacks,               //IN: array of adresses of similar patches
    const uint* __restrict g_num_patches_in_stack, //IN: numbers of patches in 3D groups
    float* __restrict patch_stack,                 //OUT: assembled 3D groups
    const uint2 image_dim,                         //IN: image dimensions
    const uint2 stacks_dim,                        //IN: dimensions limiting addresses of reference patches
    const Params params)                           //IN: denoising parameters
{
  
  
  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point,  params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);

  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;
  
  patch_stack += startidx;
  
  const ushort* z_ptr = &stacks[ idx3(0, blockIdx.x, blockIdx.y, params.N,  gridDim.x) ];

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ];
  
  patch_stack[ idx3(threadIdx.x, threadIdx.y, 0, params.k, params.k) ] = (float)(image[ idx2(outer_address.x+threadIdx.x, outer_address.y+threadIdx.y, image_dim.x)]);
  for(uint i = 0; i < num_patches; ++i)
  {
    int x = (int)((signed char)(z_ptr[i] & 0xFF));
    int y = (int)((signed char)((z_ptr[i] >> 8) & 0xFF));
    patch_stack[ idx3(threadIdx.x, threadIdx.y, i+1, params.k, params.k) ] = (float)(image[ idx2(outer_address.x+x+threadIdx.x, outer_address.y+y+threadIdx.y, image_dim.x)]);
  }
}

/*
1) Do the Walsh-Hadamard 1D transform on the z axis of 3D stack. 
2) Treshold every pixel and count the number of non-zero coefficients
3) Do the inverse Walsh-Hadamard 1D transform on the z axis of 3D stack.
Used parameters: L3D,N,k,p
Division: Each block delas with one transformed patch stack. (number of threads in block should be k*k)
*/
__global__
void hard_treshold_block(
  const uint2 start_point,                        //IN: first reference patch of a batch
  float* __restrict patch_stack,                  //IN/OUT: 3D groups with thransfomed patches
  float* __restrict  w_P,                         //OUT: weight of each 3D group
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  uint2 stacks_dim,                               //IN: dimensions limiting addresses of reference patches
  const Params params,                            //IN: denoising parameters
  const uint sigma                                //IN: noise variance
)
{
  extern __shared__ float data[];  

  int paramN = params.N+1;
  uint tcount = blockDim.x*blockDim.y;
  uint tid = idx2(threadIdx.x, threadIdx.y, blockDim.x);
  uint patch_stack_size = tcount * paramN;

  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point, patch_stack_size, stacks_dim, params, outer_address, startidx);
  
  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ]+1; //+1 for the reference patch.
  float* s_patch_stack = data + (tid * (num_patches+1)); //+1 for avoiding bank conflicts //TODO:sometimes
  patch_stack = patch_stack + startidx + tid;
    
  //Load to the shared memory
  for(uint i = 0; i < num_patches; ++i)
    s_patch_stack[i] = patch_stack[ i*tcount ];  

  //1D Transform
  fwht(s_patch_stack, num_patches);
  
  //Hard-thresholding + counting of nonzero coefficients
  uint nonzero = 0;
  float threshold = params.L3D * sqrtf((float)(num_patches * sigma));
  for(int i = 0; i < num_patches; ++i)
  {
    if (fabsf(s_patch_stack[ i ]) < threshold)
    {
      s_patch_stack[ i ] = 0.0f;
    }
    else 
      ++nonzero;
  }
  
  //Inverse 1D Transform
  fwht(s_patch_stack, num_patches);
  
  //Normalize and save to global memory
  for (uint i = 0; i < num_patches; ++i)
  {
    patch_stack[ i*tcount ] = s_patch_stack[i] / num_patches;
  }
  
  //Reuse the shared memory for 32 partial sums
  __syncthreads();
  uint* shared = (uint*)data;
  //Sum the number of non-zero coefficients for a 3D group
  nonzero = blockReduceSum<uint>(shared, nonzero, tid, tcount);
  
  //Save the weight of a 3D group (1/nonzero coefficients)
  if (tid == 0)
  {
    if (nonzero < 1) nonzero = 1;
    w_P[ idx2(blockIdx.x, blockIdx.y, gridDim.x ) ] = 1.0f/(float)nonzero;
  }
}

/*
Fills two buffers: numerator and denominator in order to compute weighted average of pixels
Used parameters: k,N,p
Division: Each block delas with one transformed patch stack.
*/
__global__
void aggregate_block(
  const uint2 start_point,                        //IN: first reference patch of a batch
  const float* __restrict patch_stack,            //IN: 3D groups with thransfomed patches
  const float* __restrict w_P,                    //IN: weight for each 3D group
  const ushort* __restrict stacks,                //IN: array of adresses of similar patches
  const float* __restrict kaiser_window,          //IN: kaiser window
  float* __restrict numerator,                    //IN/OUT: numerator aggregation buffer (have to be initialized to 0)
  float* __restrict denominator,                  //IN/OUT: denominator aggregation buffer (have to be initialized to 0)
  const uint* __restrict g_num_patches_in_stack,  //IN: numbers of patches in 3D groups
  const uint2 image_dim,                          //IN: image dimensions
  const uint2 stacks_dim,                         //IN: dimensions limiting addresses of reference patches
  const Params params                             //IN: denoising parameters
)
{    
  uint startidx;
  uint2 outer_address;
  get_block_addresses(start_point, params.k*params.k*(params.N+1), stacks_dim, params, outer_address, startidx);
  
  if (outer_address.x >= stacks_dim.x || outer_address.y >= stacks_dim.y) return;

  patch_stack += startidx;

  uint num_patches = g_num_patches_in_stack[ idx2(blockIdx.x, blockIdx.y, gridDim.x) ]+1;

  float wp = w_P[ idx2(blockIdx.x, blockIdx.y, gridDim.x ) ];
  
  const ushort* z_ptr = &stacks[ idx3(0, blockIdx.x, blockIdx.y, params.N,  gridDim.x) ];

  float kaiser_value = kaiser_window[ idx2(threadIdx.x, threadIdx.y, params.k) ];

  for(uint z = 0; z < num_patches; ++z)
  {
    int x = 0;
    int y = 0;
    if (z > 0) {
      x = (int)((signed char)(z_ptr[z-1] & 0xFF));
      y = (int)((signed char)((z_ptr[z-1] >> 8) & 0xFF));
    }

    float value = ( patch_stack[ idx3(threadIdx.x, threadIdx.y, z, params.k, params.k) ]);
    int idx = idx2(outer_address.x + x + threadIdx.x, outer_address.y + y + threadIdx.y, image_dim.x);
    atomicAdd(numerator + idx, value * kaiser_value * wp);
    atomicAdd(denominator + idx, kaiser_value * wp);
  }
}

/*
Divide numerator with denominator and round result to image_o
*/
__global__
void aggregate_final(
  const float* __restrict numerator,    //IN: numerator aggregation buffer
  const float* __restrict denominator,  //IN: denominator aggregation buffer
  const uint2 image_dim,                //IN: image dimensions
  uchar*__restrict result)              //OUT: image estimate
{
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= image_dim.x || idy >= image_dim.y) return;

  int value = lrintf(numerator[ idx2(idx,idy,image_dim.x) ] / denominator[ idx2(idx,idy,image_dim.x) ] );
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  result[ idx2(idx,idy,image_dim.x) ] = (uchar)value;
}


extern "C" void run_get_block(
  const uint2 start_point,
  const uchar* __restrict image,
  const ushort* __restrict stacks,
  const uint* __restrict num_patches_in_stack,
  float* __restrict patch_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks)
{
  get_block<<<num_blocks,num_threads>>>(
    start_point,
    image,
    stacks,
    num_patches_in_stack,
    patch_stack,
    image_dim,
    stacks_dim,
    params
  );
}

extern "C" void run_hard_treshold_block(
  const uint2 start_point,
  float* __restrict patch_stack,
  float* __restrict w_P,
  const uint* __restrict num_patches_in_stack,
  const uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size)
{
  hard_treshold_block<<<num_blocks, num_threads, shared_memory_size>>>(
    start_point,
    patch_stack,
    w_P,
    num_patches_in_stack,
    stacks_dim,
    params,
    sigma
  );
}

extern "C" void run_aggregate_block(
  const uint2 start_point,
  const float* __restrict patch_stack,  
  const float* __restrict w_P,
  const ushort* __restrict stacks,
  const float* __restrict kaiser_window,
  float* __restrict numerator,
  float* __restrict denominator,
  const uint* __restrict num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks)
{
  aggregate_block<<<num_blocks,num_threads>>>(
    start_point,
    patch_stack,
    w_P,
    stacks,
    kaiser_window,
    numerator,
    denominator,
    num_patches_in_stack,
    image_dim,
    stacks_dim,
    params
  );
}

extern "C" void run_aggregate_final(
  const float* __restrict numerator,
  const float* __restrict denominator,
  const uint2 image_dim,
  uchar*__restrict  denoised_image,
  const dim3 num_threads,
  const dim3 num_blocks
)
{
  aggregate_final<<<num_blocks,num_threads>>>(
    numerator,
    denominator,
    image_dim,
    denoised_image
  );  
}

