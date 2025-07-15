/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include "common.h"
#include <math.h>

inline void compare_output(T *outp, T *outpCPU, int size) {
  double sum_delta2, sum_ref2, L1norm2;
  sum_delta2 = 0;
  sum_ref2   = 0;
  L1norm2    = 0;
  for(int i = 0; i < size; i++) {
    sum_delta2 += std::abs(outp[i] - outpCPU[i]);
    sum_ref2 += std::abs(outpCPU[i]);
  }
  if(sum_ref2 == 0) sum_ref2 = 1;
  L1norm2      = (double)(sum_delta2 / sum_ref2);
  if(L1norm2 >= 1e-6){
    printf("Test failed\n");
  } else {
    printf("Test Passed\n");
  }
}

// Sequential implementation for comparison purposes
inline void cpu_streamcompaction(T *input, int size, int value) {
  int            pos = 0;
  // start timer
  for(int my = 0; my < size; my++) {
    if(input[my] != value) {
      input[pos] = input[my];
      pos++;
    }
  }
}

inline void verify(T *input, T *input_array, int size, int value, int size_compact) {
  cpu_streamcompaction(input_array, size, value);
  compare_output(input, input_array, size_compact);
}
