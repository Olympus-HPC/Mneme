#ifndef SOBEL_H
#define SOBEL_H

#include <chrono>
#include <cmath>
#include <cuda.h>

typedef unsigned char uchar;

void reference (uchar4 *verificationOutput,
                const uchar4 *inputImageData, 
                const uint width,
                const uint height,
                const int pixelSize);

#endif
