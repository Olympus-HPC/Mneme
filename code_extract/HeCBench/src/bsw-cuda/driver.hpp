#ifndef DRIVER_HPP
#define DRIVER_HPP

#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

#define NOW std::chrono::high_resolution_clock::now()


// for storing the alignment results
struct alignment_results
{
  short* ref_begin;
  short* query_begin;
  short* ref_end;
  short* query_end;
  short* top_scores;
};

void
kernel_driver_aa(std::string drvFile, std::vector<std::string> &reads, std::vector<std::string> &contigs, 
    short scoring_matrix[], short openGap, short extendGap);

void
verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
    short* g_alBend);
#endif
