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

inline __device__
float luminance(float r, float g, float b)
{
  return ( 0.2126f * r ) + ( 0.7152f * g ) + ( 0.0722f * b );
}

__global__
void toneMapping(
    const float *__restrict__ const input, 
          float *__restrict__ const output, 
    const float averageLuminance, 
    const float gamma, 
    const float c, 
    const float delta,
    const uint width,
    const uint numChannels,
    const uint height)
{
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  float r, g, b;
  float cLPattanaik;
  float yLPattanaik;

  float r1 = input[width * numChannels * y + (x * numChannels + 0)];
  float g1 = input[width * numChannels * y + (x * numChannels + 1)];
  float b1 = input[width * numChannels * y + (x * numChannels + 2)];

  float yLuminance = luminance(r1, g1, b1);
  float gcPattanaik = c * averageLuminance;

  if (x != 0 && y != 0 && x != width-1 && y != height-1)
  {
    //Calculating mean
    float leftUp = 0.0f;
    float up = 0.0f;
    float rightUp = 0.0f;
    float left = 0.0f;
    float right = 0.0f;
    float leftDown = 0.0f;
    float down = 0.0f;
    float rightDown = 0.0f;

    r = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 0];
    g = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 1];
    b = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 2];

    leftUp = luminance( r, g, b );

    r = input[width * numChannels * (y - 1) + ((x) * numChannels) + 0];
    g = input[width * numChannels * (y - 1) + ((x) * numChannels) + 1];
    b = input[width * numChannels * (y - 1) + ((x) * numChannels) + 2];

    up = luminance( r, g, b );

    r = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 0];
    g = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 1];
    b = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 2];

    rightUp = luminance( r, g, b );

    r = input[width * numChannels * (y) + ((x - 1) * numChannels) + 0];
    g = input[width * numChannels * (y) + ((x - 1) * numChannels) + 1];
    b = input[width * numChannels * (y) + ((x - 1) * numChannels) + 2];

    left = luminance( r, g, b );  

    r = input[width * numChannels * (y) + ((x + 1) * numChannels) + 0];
    g = input[width * numChannels * (y) + ((x + 1) * numChannels) + 1];
    b = input[width * numChannels * (y) + ((x + 1) * numChannels) + 2];

    right = luminance( r, g, b );  

    r = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 0];
    g = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 1];
    b = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 2];

    leftDown = luminance( r, g, b );

    r = input[width * numChannels * (y + 1) + ((x) * numChannels) + 0];
    g = input[width * numChannels * (y + 1) + ((x) * numChannels) + 1];
    b = input[width * numChannels * (y + 1) + ((x) * numChannels) + 2];

    down = luminance( r, g, b );

    r = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 0];
    g = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 1];
    b = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 2];

    rightDown = luminance( r, g, b );

    //Calculate median    
    yLPattanaik = (leftUp + up + rightUp + left + right + leftDown + down + rightDown) / 8;    
  }
  else
  {
    yLPattanaik = yLuminance;
  }

  cLPattanaik =  yLPattanaik * logf(delta + yLPattanaik / yLuminance) + gcPattanaik;


  float yDPattanaik = yLuminance / (yLuminance + cLPattanaik);

  r = powf((r1 / yLuminance), gamma) * yDPattanaik;
  g = powf((g1 / yLuminance), gamma) * yDPattanaik;  
  b = powf((b1 / yLuminance), gamma) * yDPattanaik;

  output[width * numChannels * y + (x * numChannels + 0)] = r;
  output[width * numChannels * y + (x * numChannels + 1)] = g;
  output[width * numChannels * y + (x * numChannels + 2)] = b;
  output[width * numChannels * y + (x * numChannels + 3)] = 
    input[width * numChannels * y + (x * numChannels + 3)];
}
