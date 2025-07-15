/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

__device__
float calNumEigenValuesLessThan(
   const float x, 
   const uint width, 
   const float *__restrict__ diagonal, 
   const float *__restrict__ offDiagonal)
{
  uint count = 0;

  float prev_diff = (diagonal[0] - x);
  count += (prev_diff < 0)? 1 : 0;
  for(uint i = 1; i < width ; i += 1)
  {
    float diff = (diagonal[i] - x) - ((offDiagonal[i-1] * offDiagonal[i-1]) / prev_diff);

    count += (diff < 0) ? 1 : 0;
    prev_diff = diff;
  }
  return count;
}

__global__
void calNumEigenValueInterval(
    uint  *__restrict__ numEigenIntervals,
    const float *__restrict__ eigenIntervals,
    const float *__restrict__ diagonal, 
    const float *__restrict__ offDiagonal,
    const uint     width)
{
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint lowerId = 2 * gid; 
  uint upperId = lowerId + 1;
  float lowerLimit = eigenIntervals[lowerId];
  float upperLimit = eigenIntervals[upperId];
  uint lower = calNumEigenValuesLessThan(lowerLimit, width, diagonal, offDiagonal);
  uint upper = calNumEigenValuesLessThan(upperLimit, width, diagonal, offDiagonal);
  numEigenIntervals[gid] = upper - lower;
}

__global__
void recalculateEigenIntervals(
          float *__restrict__ newEigenIntervals,
    const float *__restrict__ eigenIntervals,
    const uint  *__restrict__ numEigenIntervals,
    const float *__restrict__ diagonal,
    const float *__restrict__ offDiagonal,
    const    uint    width,  
    const    float   tolerance)
{
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint lowerId = 2 * gid; 
  uint upperId = lowerId + 1;
  uint currentIndex = gid;

  uint index = 0;
  while(currentIndex >= numEigenIntervals[index])
  {
    currentIndex -= numEigenIntervals[index];
    ++index;
  }

  uint lId = 2 * index;
  uint uId = lId + 1;

  /* if the number of eigenvalues in the interval is just 1 */
  if(numEigenIntervals[index] == 1)
  {
    float midValue = (eigenIntervals[uId] + eigenIntervals[lId])/2;
    float n        = calNumEigenValuesLessThan(midValue, width, diagonal, offDiagonal);
    n -= calNumEigenValuesLessThan(eigenIntervals[lId], width, diagonal, offDiagonal);

    /* check if the interval size is less than tolerance levels */
    if(eigenIntervals[uId] - eigenIntervals[lId] < tolerance)
    {
      newEigenIntervals[lowerId] = eigenIntervals[lId];
      newEigenIntervals[upperId] = eigenIntervals[uId];
    }
    else if(n == 0) /* if the eigenvalue lies in the right half of the interval */
    {
      newEigenIntervals[lowerId] = midValue;
      newEigenIntervals[upperId] = eigenIntervals[uId];
    }
    else           /* if the eigenvalue lies in the left half of the interval */
    {
      newEigenIntervals[lowerId] = eigenIntervals[lId];
      newEigenIntervals[upperId] = midValue;
    }
  }
  /* split the intervals into equal intervals of size divisionWidth */
  else /* (numEigenIntervals[index] > 1) */
  {
    float divisionWidth = (eigenIntervals[uId] - eigenIntervals[lId]) / numEigenIntervals[index];
    newEigenIntervals[lowerId] = eigenIntervals[lId] + divisionWidth * currentIndex;
    newEigenIntervals[upperId] = newEigenIntervals[lowerId] + divisionWidth;
  }  
}
