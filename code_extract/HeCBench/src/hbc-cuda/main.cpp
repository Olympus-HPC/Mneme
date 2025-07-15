#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

#include "parse.h"
#include "sequential.h"
#include "util.h"

int main(int argc, char *argv[])
{
  program_options op = parse_arguments(argc,argv);
  int max_threads_per_block, number_of_SMs;
  query_device(max_threads_per_block,number_of_SMs,op);

  graph g = parse(op.infile);

  std::cout << "Number of nodes: " << g.n << std::endl;
  std::cout << "Number of edges: " << g.m << std::endl;

  //If we're approximating, choose source vertices at random
  std::set<int> source_vertices;
  if(op.approx)
  {
    if(op.k > g.n || op.k < 1) op.k = g.n;

    while(source_vertices.size() < (size_t)op.k)
    {
      int temp_source = rand() % g.n;
      source_vertices.insert(temp_source);
    }
  }

  float CPU_time = 0;
  std::vector<float> bc;
  if(op.verify) //Only run CPU code if verifying
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    bc = bc_cpu(g,source_vertices);
    auto t2 = std::chrono::high_resolution_clock::now();
    CPU_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  }

  std::vector<float> bc_g;
  auto t1 = std::chrono::high_resolution_clock::now();
  bc_g = bc_gpu(g,max_threads_per_block,number_of_SMs,op,source_vertices);
  auto t2 = std::chrono::high_resolution_clock::now();
  float GPU_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  if(op.verify) verify(g,bc,bc_g);
  if(op.printBCscores) g.print_BC_scores(bc_g,op.scorefile);

  std::cout << std::setprecision(9);
  if(op.verify) std::cout << "Time for CPU execution: " << CPU_time/1.0e6 << " s" << std::endl;
  std::cout << "Time for GPU execution: " << GPU_time/1.0e6 << " s" << std::endl;

  delete[] g.R;
  delete[] g.C;
  delete[] g.F;

  return 0;
}
