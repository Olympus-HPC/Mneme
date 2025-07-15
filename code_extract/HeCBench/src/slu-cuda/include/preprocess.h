#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "nics_config.h"
#include "nicslu.h"
#include "type.h"

#define IN__
#define OUT__

int preprocess( \
  IN__ char *matrixName, \
  IN__ SNicsLU *nicslu,\
  OUT__ double **ax, \
  OUT__ unsigned int **ai, \
  OUT__ unsigned int **ap);

#endif
