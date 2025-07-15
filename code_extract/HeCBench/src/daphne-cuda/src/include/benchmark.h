/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef benchmark_h
#define benchmark_h

#include <iostream>

class kernel {
public:
  // performs necessary pre-run initialisation. Usually this 
  // reads the necessary input data from a file (or has them in a static array)
  virtual void init() = 0;
  
  // executes the testcase, in blocks of p at a time
  virtual void run(int p = 1) = 0;
  
  // compares the computed output with the golden reference output
  // output can be read from a file or in a static array
  // for floating point results a given error constant should be given
  virtual bool check_output() = 0;
  
  // number of testcase available for this kernel (there should be at least 1)
  int testcases = 1;
  
  // sets the functions which should be called to pause and unpause the timer
  void set_timer_functions(void (*pause_function)(),
		            void (*unpause_function)()) {
     unpause_func = unpause_function;
     pause_func = pause_function;
   }
  
protected:
  void (*unpause_func)();
  void (*pause_func)();
  virtual int read_next_testcases(int count) = 0;
};

#endif
