//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _REDUCTION_H_
#define _REDUCTION_H_

/* warp reduction with shfl function */
template < typename T >
__inline__ __device__ float warp_reduce(T val)
{
  for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

/* block reduction with warp reduction */
template < typename T >
__inline__ __device__ float block_reduce(T val)
{
  static __shared__ T shared[WARPSIZE];
  int tid = threadIdx.z * BY * BX + threadIdx.y * BX + threadIdx.x;
  int lane = tid & (WARPSIZE-1);
  int wid = tid/WARPSIZE;
  val = warp_reduce<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (tid < (blockDim.x * blockDim.y * blockDim.z)/WARPSIZE) ? shared[lane] : 0;
  if(wid == 0){
    val = warp_reduce<T>(val);
  }
  return val;
}


/* energy reduction using block reduction */
template <typename T>
__global__ void kernel_redenergy(const int *s, int L, T *out, const int *H, float h)
{
  // offsets
  int x = blockIdx.x *blockDim.x + threadIdx.x;
  int y = blockIdx.y *blockDim.y + threadIdx.y;
  int z = blockIdx.z *blockDim.z + threadIdx.z;
  int tid = threadIdx.z * BY * BX + threadIdx.y * BX + threadIdx.x;
  int id = C(x,y,z,L);
  // this optimization only works for L being a power of 2
  //float sum = -(float)(s[id] * ((float)(s[C((x+1) & (L-1), y, z, L)] + 
  // s[C(x, (y+1) & (L-1), z, L)] + s[C(x, y, (z+1) & (L-1), L)]) + h*H[id]));

  // this line works always
  float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] + 
              s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));
  sum = block_reduce<T>(sum); 

  if(tid == 0) atomicAdd(out, sum);
}


#endif
