#include "hwt.h"


void calApproxFinalOnHost(float *inData, float *hOutData, unsigned int signalLength)
{
  float *tempOutData = (float*)malloc(signalLength * sizeof(float));

  memcpy(tempOutData, inData, signalLength * sizeof(float));

  for(unsigned int i = 0; i < signalLength; ++i)
  {
    tempOutData[i] = tempOutData[i] / sqrt((float)signalLength);
  }

  unsigned int length = signalLength;
  while(length > 1u)
  {
    for(unsigned int i = 0; i < length / 2; ++i)
    {
      float data0 = tempOutData[2 * i];
      float data1 = tempOutData[2 * i + 1];

      hOutData[i] = (data0 + data1) / sqrtf(2.f);
      hOutData[length / 2 + i] = (data0 - data1) / sqrtf(2.f);
    }
    memcpy(tempOutData, hOutData, signalLength * sizeof(float));
    length >>= 1;
  }

  free(tempOutData);
}

int getLevels(unsigned int length, unsigned int* levels)
{
  int returnVal = 1;

  for(unsigned int i = 0; i < 24; ++i)
  {
    if(length == (1U << i))
    {
      *levels = i;
      returnVal = 0;
      break;
    }
  }

  return returnVal;
}

