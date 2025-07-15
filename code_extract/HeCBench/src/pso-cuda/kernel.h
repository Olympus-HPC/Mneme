#ifndef KERNEL_H_
#define KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define F(x) (1.f + ((x) - 1.f) / 4.f)

const   int DIM = 30;  // number of dimensions
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float OMEGA = 0.5f;
const float c1 = 1.5f;
const float c2 = 1.5f;
const float phi = 3.1415f;

float getRandom(float low,float high);
float getRandomClamped(int seed);
float host_fitness_function(float x[]);

void pso(int p,int r,float *positions,float *velocities,float *pBests,float *gBest);
extern "C" void gpu_pso(int p,int r,float *positions,float *velocities,float *pBests,float *gBest);

#endif /* KERNEL_H_ */
