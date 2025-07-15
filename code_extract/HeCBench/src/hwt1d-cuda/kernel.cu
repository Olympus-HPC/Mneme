/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 */


/**
 * @brief   Calculates decomposed signal with maximum of 9 levels by using 
 *          1D Haar wavelet decomposition 
 * @param   inSignal        input signal
 * @param   coefsSignal     Coefficient details of signal after 9 levels of decompostion 
 * @param   AverageSignal   Averages of signal after 9 levels of decompostion
 * @param   sharedArray     shared array 
 * @param   tLevels         actual levels required for full decomposition 
 * @param   signalLength    length of signal
 * @param   levelsDone      level of decompositions done 
 * @param   mLevels	    maximum number of levels to be processed on device
 */

#include "hwt.h"

__global__
void dwtHaar1D( const float *__restrict__ inSignal,
                      float *__restrict__ coefsSignal,
                      float *__restrict__ AverageSignal,
                      const unsigned int tLevels,
                      const unsigned int signalLength,
                      const unsigned int levelsDone,
		      const unsigned int mLevels)
              
{
    size_t localId = threadIdx.x;
    size_t groupId = blockIdx.x;
    size_t localSize = blockDim.x;

    extern __shared__ float sharedArray[];
    
    /**
     * Read input signal data from global memory
     * to shared memory
     */
    float t0 = inSignal[groupId * localSize * 2 + localId];
    float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
    // Divide with signal length for normalized decomposition
    if(0 == levelsDone)
    {
       float r = rsqrtf((float)signalLength);
       t0 *= r;
       t1 *= r;
    }
    sharedArray[localId] = t0;
    sharedArray[localSize + localId] = t1;
     
    __syncthreads();
    
    unsigned int levels = tLevels > mLevels ? mLevels: tLevels;
    unsigned int activeThreads = (1 << levels) / 2;
    unsigned int midOutPos = signalLength / 2;
    
    const float rsqrt_two = 0.7071f;
    for(unsigned int i = 0; i < levels; ++i)
    {

        float data0, data1;
        if(localId < activeThreads)
        {
            data0 = sharedArray[2 * localId];
            data1 = sharedArray[2 * localId + 1];
        }

        /* make sure all work items have read from sharedArray before modifying it */
        __syncthreads();

        if(localId < activeThreads)
        {
            sharedArray[localId] = (data0 + data1) * rsqrt_two;
            unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
            coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;
       
            midOutPos >>= 1;
        }
        activeThreads >>= 1;
        __syncthreads();
    }
    
    /**
     * Write 0th element for the next decomposition
     * steps which are performed on host 
     */
    
     if(0 == localId)
        AverageSignal[groupId] = sharedArray[0];
}

int runKernel(
    float *inData, 
    float *dOutData, 
    float *hOutData, 
    float *dPartialOutData,
    const unsigned int signalLength,
    float *inDataBuf,
    float *partialOutDataBuf,
    float *outDataBuf)
{
  unsigned int levels = 0;

  int result = getLevels(signalLength, &levels);
  if (result == 1) {
    std::cerr << "getLevels() failed\n";
    return 1;
  }

  unsigned int actualLevels = levels;

  //max levels on device should be decided by kernelWorkGroupSize
  //int tempVar = (int)(log((float)kernelInfo.kernelWorkGroupSize) / log((float)2));
  //int maxLevelsOnDevice = tempVar + 1;
  // Assume work group size is 256
  const int maxLevelsOnDevice = 9;

  const int signalLengthByte = signalLength * sizeof(float);
  float* temp = (float*)malloc(signalLengthByte);
  memcpy(temp, inData, signalLengthByte);

  int levelsDone = 0;
  int one = 1;
  unsigned int curLevels = 0;
  unsigned int curSignalLength;
  while((unsigned int)levelsDone < actualLevels)
  {
    curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

    // Set the signal length for current iteration
    if(levelsDone == 0)
    {
      curSignalLength = signalLength;
    }
    else
    {
      curSignalLength = (one << levels);
    }

    // Set group size
    unsigned int groupSize = (1 << curLevels) / 2;

    unsigned int totalLevels = levels;

    cudaMemcpy(inDataBuf, inData, signalLengthByte, cudaMemcpyHostToDevice);

    dim3 grid ((curSignalLength >> 1) / groupSize);
    dim3 block (groupSize);

    dwtHaar1D<<<grid, block, groupSize*2*sizeof(float)>>>(
      inDataBuf,
      outDataBuf,
      partialOutDataBuf,
      totalLevels,
      curSignalLength,
      levelsDone,
      maxLevelsOnDevice);

    cudaMemcpy(dOutData, outDataBuf, signalLengthByte, cudaMemcpyDeviceToHost);

    cudaMemcpy(dPartialOutData, partialOutDataBuf, signalLengthByte, cudaMemcpyDeviceToHost);

    if(levels <= maxLevelsOnDevice)
    {
      dOutData[0] = dPartialOutData[0];
      memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
      memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
          (signalLength  - (one << curLevels)) * sizeof(float));
      break;
    }
    else
    {
      levels -= maxLevelsOnDevice;
      memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
      memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
      levelsDone += (int)maxLevelsOnDevice;
    }
  }

  memcpy(inData, temp, signalLengthByte);
  free(temp);
  return 0;
}
