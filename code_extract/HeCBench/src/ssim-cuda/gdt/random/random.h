// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* pieces originally taken from optixPathTracer/random.h example,
   under following license */

/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gdt/gdt.h"
#include "gdt/math/vec.h"

namespace gdt {

///////////////////////////////////////////////////////////////////////////////
// TEA - Random numbers based on Tiny Encryption Algorithm ////////////////////
///////////////////////////////////////////////////////////////////////////////
// https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/intro_driver/shaders/random_number_generators.h

// struct RandomTEA {
// 
//   __device__ RandomTEA(const unsigned int idx, const unsigned int seed)
//   {
//     this->v0 = idx;
//     this->v1 = seed;
//   }
// 
//   __device__ float2 get_floats()
//   {
//     tea8(this->v0, this->v1);
//     const float tofloat = 2.3283064365386962890625e-10f; // 1/2^32
//     return make_float2(this->v0 * tofloat, this->v1 * tofloat);
//   }
// 
// private:
//   inline __device__ void tea8(unsigned int& _v0, unsigned int& _v1)
//   {
//     unsigned int v0 = _v0; // Operate on registers to avoid slowdown!
//     unsigned int v1 = _v1;
//     unsigned int sum = 0;
// 
//     for (int i = 0; i < 8; i++) { // just 8 instead of 32 rounds
//       sum += 0x9e3779b9;
//       v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
//       v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
//     }
// 
//     _v0 = v0;
//     _v1 = v1;
//   }
// 
//   unsigned int v0, v1;
// };

  /*! simple 24-bit linear congruence generator */
  template<unsigned int N=16>
  struct LCG {
    
    inline __both__ LCG()
    { /* intentionally empty so we can use it in device vars that
         don't allow dynamic initialization (ie, PRD) */
    }

    inline __both__ LCG(unsigned int val0, unsigned int val1) { init(val0,val1); }
    
    inline __both__ void init(unsigned int val0, unsigned int val1)
    {
      unsigned int v0 = val0;
      unsigned int v1 = val1;
      unsigned int s0 = 0;
      
      for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
      }
      state = v0;
    }
    
    // Generate random unsigned int in [0, 2^24)
    inline __both__ float operator() ()
    {
      return get_float();
    }

    inline __both__ float get_float()
    {
      const uint32_t LCG_A = 1664525u;
      const uint32_t LCG_C = 1013904223u;
      state = (LCG_A * state + LCG_C);
      return (state & 0x00FFFFFF) / (float) 0x01000000;
    }

    inline __both__ vec2f get_floats()
    {
      return vec2f(get_float(), get_float());
    }

    uint32_t state;
  };

} // ::gdt
