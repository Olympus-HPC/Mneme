#ifndef BC_UTIL
#define BC_UTIL

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include "parse.h"

//Command line parsing
class program_options
{
  public:
    program_options() : infile(NULL), verify(false), printBCscores(false), 
                        scorefile(NULL), device(-1), approx(false), k(256) {}

    char *infile;
    bool verify;
    bool printBCscores;
    char *scorefile;
    int device;
    bool approx;
    int k;
};

program_options parse_arguments(int argc, char *argv[]);

void query_device(int &max_threads_per_block, int &number_of_SMs, program_options op);

// compare cpu and gpu results
void verify(graph g, const std::vector<float> bc_cpu, const std::vector<float> bc_gpu);

// run bc on a GPU device
std::vector<float> bc_gpu(
  graph g,
  int max_threads_per_block,
  int number_of_SMs,
  program_options op,
  const std::set<int> &source_vertices);

#endif

