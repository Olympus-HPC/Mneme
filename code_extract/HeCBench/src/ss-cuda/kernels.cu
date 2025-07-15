/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  • Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  • Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

/**
 * @brief Compare two strings with specified length
 * @param text       start position on text string
 * @param pattern    start position on pattern string
 * @param length     Length to compare
 * @return 0-failure, 1-success
 */
__device__
int compare(const uchar* text, const uchar* pattern, uint length)
{
  for(uint l=0; l<length; ++l)
  {
    if (TOLOWER(text[l]) != pattern[l]) return 0;
  }
  return 1;
}

/**
 * @brief Naive kernel version of string search.
 *        Find all pattern positions in the given text
 * @param text               Input Text
 * @param textLength         Length of the text
 * @param pattern            Pattern string
 * @param patternLength      Pattern length
 * @param resultBuffer       Result of all matched positions
 * @param resultCountPerWG   Result counts per Work-Group
 * @param maxSearchLength    Maximum search positions for each work-group
 * @param localPattern       local buffer for the search pattern
 */
__global__ void 
StringSearchNaive (
    const uchar* text,
    const uint textLength,
    const uchar* pattern,
    const uint patternLength,
    uint* resultBuffer,
    uint* resultCountPerWG,
    const uint maxSearchLength)
{
  extern __shared__ uchar localPattern[];
  __shared__ uint groupSuccessCounter;

  int localIdx = threadIdx.x;
  int localSize = blockDim.x;
  int groupIdx = blockIdx.x;

  // Last search idx for all work items
  uint lastSearchIdx = textLength - patternLength + 1;

  // global idx for all work items in a WorkGroup
  uint beginSearchIdx = groupIdx * maxSearchLength;
  uint endSearchIdx = beginSearchIdx + maxSearchLength;
  if(beginSearchIdx > lastSearchIdx) return;
  if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;

  // Copy the pattern from global to local buffer
  for(int idx = localIdx; idx < patternLength; idx+=localSize)
  {
    localPattern[idx] = TOLOWER(pattern[idx]);
  }

  if(localIdx == 0) groupSuccessCounter = 0;
  __syncthreads();

  // loop over positions in global buffer
  for(uint stringPos=beginSearchIdx+localIdx; stringPos<endSearchIdx; stringPos+=localSize)
  {
    if (compare(text+stringPos, localPattern, patternLength) == 1)
    {
      int count = atomicAdd(&groupSuccessCounter, (uint)1);
      resultBuffer[beginSearchIdx+count] = stringPos;
    }
  }

  __syncthreads();
  if(localIdx == 0) resultCountPerWG[groupIdx] = groupSuccessCounter;
}

/**
 * @brief Load-Balance kernel version of string search.
 *        Find all pattern positions in the given text
 * @param text               Input Text
 * @param textLength         Length of the text
 * @param pattern            Pattern string
 * @param patternLength      Pattern length
 * @param resultBuffer       Result of all matched positions
 * @param resultCountPerWG   Result counts per Work-Group
 * @param maxSearchLength    Maximum search positions for each work-group
 * @param localPattern       local buffer for the search pattern
 * @param stack1             local stack for store initial 2-byte match 
 * @param stack2             local stack for store initial 10-byte match positions
 */
__global__ void 
StringSearchLoadBalance (
    const uchar* text,
    const uint textLength,
    const uchar* pattern,
    const uint patternLength,
    uint* resultBuffer,
    uint* resultCountPerWG,
    const uint maxSearchLength)
{
  extern __shared__ uchar localPattern[];
  __shared__ uint stack1[LOCAL_SIZE*2];
  __shared__ uint stack2[LOCAL_SIZE*2];
  __shared__ uint stack1Counter;
  __shared__ uint stack2Counter;
  __shared__ uint groupSuccessCounter;

  int localIdx = threadIdx.x;
  int localSize = blockDim.x;
  int groupIdx = blockIdx.x;

  // Initialize the local variaables
  if(localIdx == 0)
  {
    groupSuccessCounter = 0;
    stack1Counter = 0;
    stack2Counter = 0;
  }

  // Last search idx for all work items
  uint lastSearchIdx = textLength - patternLength + 1;
  uint stackSize = 0;

  // global idx for all work items in a WorkGroup
  uint beginSearchIdx = groupIdx * maxSearchLength;
  uint endSearchIdx = beginSearchIdx + maxSearchLength;
  if(beginSearchIdx > lastSearchIdx) return;
  if(endSearchIdx > lastSearchIdx) endSearchIdx = lastSearchIdx;
  uint searchLength = endSearchIdx - beginSearchIdx;

  // Copy the pattern from global to local buffer
  for(uint idx = localIdx; idx < patternLength; idx+=localSize)
  {
    localPattern[idx] = TOLOWER(pattern[idx]);
  }

  __syncthreads();

  uchar first = localPattern[0];
  uchar second = localPattern[1];
  int stringPos = localIdx;
  int stackPos = 0;
  int revStackPos = 0;

  while (true)    // loop over positions in global buffer
  {

    // Level-1 : Quick filter on 2 char match and store the good positions on stack1.
    if(stringPos < searchLength)
    {
      // Queue the initial match positions. Make sure queue has sufficient positions for each work-item.
      if ((first == TOLOWER(text[beginSearchIdx+stringPos])) && (second == TOLOWER(text[beginSearchIdx+stringPos+1])))
      {
        stackPos = atomicAdd(&stack1Counter, (uint)1);
        stack1[stackPos] = stringPos;
      }
    }

    stringPos += localSize;     // next search idx

    __syncthreads();
    stackSize = stack1Counter;
    __syncthreads();

    // continue until stack1 has sufficient good positions for proceed to next Level
    if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;


#ifdef ENABLE_2ND_LEVEL_FILTER
    // Level-2 : (Processing the stack1 and filling the stack2) For large patterns roll over
    // another 8-bytes from the positions in stack1 and store the match positions in stack2.
    if(localIdx < stackSize)
    {
      revStackPos = atomicSub(&stack1Counter, (uint)1);
      uint pos = stack1[--revStackPos];
      bool status = (localPattern[2] == TOLOWER(text[beginSearchIdx+pos+2]));
      status = status && (localPattern[3] == TOLOWER(text[beginSearchIdx+pos+3]));
      status = status && (localPattern[4] == TOLOWER(text[beginSearchIdx+pos+4]));
      status = status && (localPattern[5] == TOLOWER(text[beginSearchIdx+pos+5]));
      status = status && (localPattern[6] == TOLOWER(text[beginSearchIdx+pos+6]));
      status = status && (localPattern[7] == TOLOWER(text[beginSearchIdx+pos+7]));
      status = status && (localPattern[8] == TOLOWER(text[beginSearchIdx+pos+8]));
      status = status && (localPattern[9] == TOLOWER(text[beginSearchIdx+pos+9]));

      if (status)
      {
        stackPos = atomicAdd(&stack2Counter, (uint)1);
        stack2[stackPos] = pos;
      }
    }

    __syncthreads();
    stackSize = stack2Counter;
    __syncthreads();

    // continue until stack2 has sufficient good positions proceed to next level
    if((stackSize < localSize) && ((((stringPos)/localSize)*localSize) < searchLength)) continue;
#endif


    // Level-3 : (Processing stack1/stack2) Check the remaining positions.
    if(localIdx < stackSize)
    {
#ifdef ENABLE_2ND_LEVEL_FILTER
      revStackPos = atomicSub(&stack2Counter, (uint)1);
      int pos = stack2[--revStackPos];
      if (compare(text+beginSearchIdx+pos+10, localPattern+10, patternLength-10) == 1)
#else
      revStackPos = atomicSub(&stack1Counter, (uint)1);
      int pos = stack1[--revStackPos];
      if (compare(text+beginSearchIdx+pos+2, localPattern+2, patternLength-2) == 1)
#endif
      {
        // Full match found
        int count = atomicAdd(&groupSuccessCounter, (uint)1);
        resultBuffer[beginSearchIdx+count] = beginSearchIdx+pos;
      }
    }

    __syncthreads();
    if((((stringPos/localSize)*localSize) >= searchLength) && 
        (stack1Counter <= 0) && (stack2Counter <= 0)) break;
  }

  if(localIdx == 0) resultCountPerWG[groupIdx] = groupSuccessCounter;
}
