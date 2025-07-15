#ifndef _CCS_H_
#define _CCS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>

#define SUCCESS                  0
#define TRUE                     1
#define FALSE                    0
#define FAIL                    -1

#define mingene   10    //minimum number of genes in a bicluster 
#define minsample 10    //minimum number of samples in a bicluster
#define MAXB      1000  //default number of base gene (outer loop) in the decreasing order of SD to consider for forming the biclusters. Overide if -m number>1000 is used 


#define BUFFER_MAXSIZE 1048576


struct gn{
  char *id;
  int indx;
  float *x;
};

struct bicl{
  char *sample,*data;
  int samplecount,datacount; 
  float score;
};

struct pair_r{
 float r,n_r;
};

void printUsage();

#endif  // _CCS_H_
