/*
 * (c) 2009-2010 Christoph Schied <Christoph.Schied@uni-ulm.de>
 *
 * This file is part of flame.
 *
 * flame is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * flame is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with flame.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <cassert>
//#include <vector_types.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include "flame.hpp"

unsigned mersenne_twister(unsigned *mersenne_state)
{
  const int N = 624;
  const int M = 397;
  const unsigned A[2] = { 0, 0x9908b0df };
  const unsigned HI = 0x80000000;
  const unsigned LO = 0x7fffffff;
  static int index = N+1;

  if (index >= N) {
    unsigned h;
    for (int k=0 ; k<N-M ; ++k) {
      h = (mersenne_state[k] & HI) | (mersenne_state[k+1] & LO);
      mersenne_state[k] = mersenne_state[k+M] ^ (h >> 1) ^ A[h & 1];
    }
    for (int k=N-M ; k<N-1 ; ++k) {
      h = (mersenne_state[k] & HI) | (mersenne_state[k+1] & LO);
      mersenne_state[k] = mersenne_state[k+(M-N)] ^ (h >> 1) ^ A[h & 1];
    }
    h = (mersenne_state[N-1] & HI) | (mersenne_state[0] & LO);
    mersenne_state[N-1] = mersenne_state[M-1] ^ (h >> 1) ^ A[h & 1];
    index = 0;
  }

  unsigned e = mersenne_state[index++];
  // tempering:
  e ^= (e >> 11);
  e ^= (e << 7) & 0x9d2c5680;
  e ^= (e << 15) & 0xefc60000;
  e ^= (e >> 18);
  return e;
}

float radical_inverse(unsigned int n, unsigned int base)
{
  float res = 0;
  float div = 1.0f / float(base);

  while(n > 0) {
    float digit = n % base;
    res = res + digit * div;
    n = ( n - digit ) / base;
    div /= (float)base;
  }

  return res;
}

void check_cuda_error(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg,
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

