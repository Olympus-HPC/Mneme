/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, and Ari Harju

    This file is part of GPUQT.

    GPUQT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUQT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUQT.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
    The main function of the LSQT code
------------------------------------------------------------------------------*/

#include "lsqt.h"
#include <fstream>
#include <iostream>

static void print_welcome();
static void check_argc(int);
static void print_start(std::string);
static void print_finish(std::string, real);

int main(int argc, char* argv[])
{
  print_welcome();
  check_argc(argc);
  std::ifstream input(argv[1]); // input = the driver input file
  if (!input.is_open()) {
    std::cout << "Failed to open " << argv[1] << std::endl;
    exit(1);
  }
  std::string directory;
  while (getline(input, directory)) {
    if (directory == "") {
      continue;
    }
    print_start(directory);
    auto time_begin = std::chrono::steady_clock::now();
    lsqt(directory);
    auto time_finish = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_finish - time_begin).count();
    real time_used = time * 1e-9f;
    print_finish(directory, time_used);
  }
  return 0;
}

static void print_welcome()
{
  std::cout << std::endl;
  std::cout << "***************************************************************\n";
  std::cout << "*                  Welcome to use LSQT                        *\n";
  std::cout << "*          (Linear Scaling Quantum Transport)                 *\n";
  std::cout << "*        (Author:  Zheyong Fan <brucenju@gmail.com>)          *\n";
  std::cout << "***************************************************************\n";
  std::cout << std::endl;
}

static void check_argc(int argc)
{
  if (argc != 2) {
    std::cout << "Usage: src/gpuqt input.txt" << std::endl;
    exit(1);
  }
}

static void print_start(std::string directory)
{
  std::cout << std::endl;
  std::cout << "===============================================================\n";
  std::cout << "Run LSQT simulation for " << directory << std::endl;
  std::cout << "===============================================================\n";
}

static void print_finish(std::string directory, real time)
{
  std::cout << std::endl;
  std::cout << "===============================================================\n";
  std::cout << "Total time used for " << directory << " = " << time << " s" << std::endl;
  std::cout << "===============================================================\n";
}
