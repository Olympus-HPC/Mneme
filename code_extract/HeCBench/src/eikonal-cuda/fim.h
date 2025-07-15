//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//
#ifndef __FIM_H__
#define __FIM_H__

#include <cstdlib>
#include "common_def.h"

#define TIMER

void runEikonalSolverSimple(GPUMEMSTRUCT &cmem);

#endif
