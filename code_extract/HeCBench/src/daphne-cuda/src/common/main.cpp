/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#include <chrono>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "benchmark.h"

std::chrono::high_resolution_clock::time_point start,end;
std::chrono::duration<double> elapsed;
std::chrono::high_resolution_clock timer;
bool pause = false;

// how many testcases should be executed in sequence (before checking for correctness)
int pipelined = 1;

extern kernel& myKernel;

void pause_timer()
{
  end = timer.now();
  elapsed += (end-start);
  pause = true;
}  

void unpause_timer() 
{
  pause = false;
  start = timer.now();
}

void usage(char *exec)
{
  std::cout << "Usage: \n" << exec << " [-p N]\nOptions:\n  -p N   executes N invocations in sequence,";
  std::cout << "before taking time and check the result.\n";
  std::cout << "         Default: N=1\n";
}
int main(int argc, char **argv) {

  if ((argc != 1) && (argc !=  3))
  {
    usage(argv[0]);
    exit(2);
  }
  if (argc == 3)
  {
    if (strcmp(argv[1], "-p") != 0)
    {
      usage(argv[0]);
      exit(3);
    }
    errno = 0;
    pipelined = strtol(argv[2], NULL, 10);
    if (errno || (pipelined < 1) )
    {
      usage(argv[0]);
      exit(4);
    }
    std::cout << "Invoking kernel " << pipelined << " time(s) per measure/checking step\n";
  }
  // read input data
  myKernel.set_timer_functions(pause_timer, unpause_timer);
  myKernel.init();

  // measure the runtime of the kernel
  start = timer.now();

  // execute the kernel
  myKernel.run(pipelined);

  // measure the runtime of the kernel
  if (!pause) 
  {
    end = timer.now();
    elapsed += end-start;
  }
  std::cout << "Elapsed time: "<< elapsed.count() << " seconds, average time per testcase (#"
            << myKernel.testcases << "): " << elapsed.count() / (double) myKernel.testcases
            << " seconds" << std::endl;

  // read the desired output  and compare
  if (myKernel.check_output())
    std::cout << "PASS\n";
  else 
    std::cout << "FAIL\n";

  return 0;
}
