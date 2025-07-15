/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <chrono>
#include <iostream>
#include <map>
#include <string>

struct Timer {

  std::map<const std::string, std::chrono::steady_clock::time_point> startTime;
  std::map<const std::string, std::chrono::steady_clock::time_point> stopTime;
  std::map<const std::string, double> time;

  void start(const std::string &name) {
    if(!time.count(name)) {
      time[name] = 0.0;
    }
    startTime[name] = std::chrono::steady_clock::now();
  }

  void stop(const std::string &name) {
    stopTime[name] = std::chrono::steady_clock::now();
    float part_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime[name] - startTime[name]).count();
    time[name] += part_time;
  }

  void print(const std::string &name, const unsigned int REP) {
    printf("%s time (ms): %f\n", name.c_str(), time[name] * 1e-6f / REP);
  }
};

