/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/


#ifndef _SCANLARGEARRAYS_H_
#define _SCANLARGEARRAYS_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define GROUP_SIZE 256

template<typename T>
int isPowerOf2(T val)
{
  long long _val = val;
  if((_val & (-_val))-_val == 0 && _val != 0)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

/**
 * roundToPowerOf2
 * rounds to a power of 2
 */
template<typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for(int i = 0; i < bytes; i++)
  {
    val |= val >> (1<<i);
  }
  val++;
  return val;
}

/**
 * fillRandom
 * fill array with random values
 */
template<typename T>
int fillRandom(
    T * arrayPtr,
    const int width,
    const int height,
    const T rangeMin,
    const T rangeMax,
    unsigned int seed=123)
{
  if(!arrayPtr)
  {
    std::cout << "Cannot fill array. NULL pointer." << std::endl;
    return -1;
  }
  if(!seed)
  {
    seed = (unsigned int)time(NULL);
  }
  srand(seed);
  double range = double(rangeMax - rangeMin) + 1.0;
  /* random initialisation of input */
  for(int i = 0; i < height; i++)
    for(int j = 0; j < width; j++)
    {
      int index = i*width + j;
      arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
    }
  return 0;
}

template<typename T>
bool compare(const T *refData, const T *data, 
             const int length, const T epsilon = 1e-6)
{
  T error = 0.0;
  T ref = 0.0;
  for(int i = 1; i < length; ++i)
  {
    T diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  T normRef = std::sqrt((T) ref);
  if (std::fabs((T) ref) < epsilon)
  {
    return false;
  }
  T normError = std::sqrt((T) error);
  error = normError / normRef;
  return error < epsilon;
}


#endif //_SCANLARGEARRAYS_H_
