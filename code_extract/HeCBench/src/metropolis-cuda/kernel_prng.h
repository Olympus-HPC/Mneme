//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
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
#ifndef _KERNEL_PRNG_SETUP_
#define _KERNEL_PRNG_SETUP_


/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

#include <limits.h>
#include <inttypes.h>

#define INV_UINT_MAX 2.3283064e-10f

__host__ __device__ 
inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc);

__host__ __device__
inline void gpu_pcg32_srandom_r(uint64_t *state, uint64_t *inc, uint64_t initstate, uint64_t initseq)
{
  *state = 0U;
  *inc = (initseq << 1u) | 1u;
  gpu_pcg32_random_r(state, inc);
  *state += initstate;
  gpu_pcg32_random_r(state, inc);
}


__host__ __device__
inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc)
{
  uint64_t oldstate = *state;
  *state = oldstate * 6364136223846793005ULL + *inc;
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__host__ __device__
inline float gpu_rand01(uint64_t *state, uint64_t *inc)
{
  return (float) gpu_pcg32_random_r(state, inc) * INV_UINT_MAX;
}




// Murmur hash 64-bit
__device__
uint64_t mmhash64( const void * key, int len, unsigned int seed )
{
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

  while(data != end){
    uint64_t k = *data++;

    k *= m; 
    k ^= k >> r; 
    k *= m; 

    h ^= k;
    h *= m; 
  }
  const unsigned char * data2 = (const unsigned char*)data;
  switch(len & 7)
  {
    case 7: h ^= uint64_t(data2[6]) << 48;
    case 6: h ^= uint64_t(data2[5]) << 40;
    case 5: h ^= uint64_t(data2[4]) << 32;
    case 4: h ^= uint64_t(data2[3]) << 24;
    case 3: h ^= uint64_t(data2[2]) << 16;
    case 2: h ^= uint64_t(data2[1]) << 8;
    case 1: h ^= uint64_t(data2[0]);
      h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
} 

__global__
void kernel_gpupcg_setup(uint64_t *state, uint64_t *inc, int N, 
                         uint64_t seed, uint64_t seq)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if( x < N ){
    // exclusive seeds, per replica sequences 
    uint64_t tseed = x + seed;
    uint64_t hseed = mmhash64(&tseed, sizeof(uint64_t), 17);
    uint64_t hseq = mmhash64(&seq, sizeof(uint64_t), 47);
    gpu_pcg32_srandom_r(&state[x], &inc[x], hseed, hseq);
  }
}
#endif

