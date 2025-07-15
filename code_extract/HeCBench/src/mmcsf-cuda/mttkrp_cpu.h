#ifndef MTTKRP_CPU_H
#define MTTKRP_CPU_H

#include "util.h"

// MTTKRP on CPU using COO
void MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt);

void MTTKRP_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt);

 
#endif
