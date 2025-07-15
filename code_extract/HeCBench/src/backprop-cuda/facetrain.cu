#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;

void backprop_face()
{
  BPNN *net;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("\nFinish the training for one iteration\n");
}

void setup(int argc, char **argv)
{
  int seed;

  if (argc!=2){
    fprintf(stderr, "Usage: %s <number of input nodes>\n", argv[0]);
    exit(-1);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
    fprintf(stderr, "The number of input nodes must be divided by 16\n");
    exit(-1);
  }

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();
}
