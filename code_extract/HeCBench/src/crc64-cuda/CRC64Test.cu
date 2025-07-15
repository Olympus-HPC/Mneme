// *****************************************************************************
//                   Copyright (C) 2014, UChicago Argonne, LLC
//                              All Rights Reserved
// 	       High-Performance CRC64 Library (ANL-SF-14-095)
//                    Hal Finkel, Argonne National Laboratory
// 
//                              OPEN SOURCE LICENSE
// 
// Under the terms of Contract No. DE-AC02-06CH11357 with UChicago Argonne, LLC,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor
//    the names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission. 
//  
// *****************************************************************************
//                                  DISCLAIMER
// 
// THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
// 
// NEITHER THE UNTED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
// ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY
// WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY
// FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA,
// APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
// INFRINGE PRIVATELY OWNED RIGHTS.
// 
// *****************************************************************************
#include <ctime>
#include <vector>
#include <iostream>
#include "CRC64.h"

int main(int argc, char *argv[]) {
  int ntests = 10;
  if (argc > 1) ntests = atoi(argv[1]);

  int seed = 5;
  if (argc > 2) seed = atoi(argv[2]);

  int max_test_length = 2097152;
  if (argc > 3) max_test_length = atoi(argv[3]);

  std::cout << "Running " << ntests << " tests with seed " << seed << std::endl;

  srand48(seed);

#ifdef __bgp__
#define THE_CLOCK CLOCK_REALTIME
#else
#define THE_CLOCK CLOCK_THREAD_CPUTIME_ID
#endif

  double tot_time = 0, tot_bytes = 0;

  int ntest = 0;
  while (++ntest <= ntests) {
    std::cout << ntest << " ";

    size_t test_length = (size_t) (max_test_length*(drand48()+1));
    std::cout << test_length << " ";

    std::vector<unsigned char> input_buffer(test_length);

    for (size_t i = 0; i < test_length; ++i) {
      input_buffer[i] = (unsigned char) (255*drand48());
    }

    timespec b_start, b_end;
    clock_gettime(THE_CLOCK, &b_start);

    uint64_t cs = crc64_parallel(&input_buffer[0], test_length);

    clock_gettime(THE_CLOCK, &b_end);
    double b_time = (b_end.tv_sec - b_start.tv_sec);
    b_time += 1e-9*(b_end.tv_nsec - b_start.tv_nsec);

    if (ntest > 1) {
      tot_time += b_time;
      tot_bytes += test_length;
    }

    // Copy the input_buffer and append the check bytes.
    size_t tlend = 8;
    input_buffer.resize(test_length + tlend, 0);
    crc64_invert(cs, &input_buffer[test_length]);

    std::string pass("pass"), fail("fail");
    uint64_t csc = crc64(&input_buffer[0], test_length+tlend);
    std::cout << ((csc == (uint64_t) -1) ? pass : fail) << " ";

    size_t div_pt = (size_t) (test_length*drand48());
    uint64_t cs1 = crc64(&input_buffer[0], div_pt);
    uint64_t cs2 = crc64(&input_buffer[div_pt], test_length - div_pt);
    csc = crc64_combine(cs1, cs2, test_length - div_pt);
    std::cout << ((csc == cs) ? pass : fail);

    std::cout << std::endl;
  }

  std::cout << (tot_bytes/(1024*1024))/tot_time << " MB/s" << std::endl;

  return 0;
}
