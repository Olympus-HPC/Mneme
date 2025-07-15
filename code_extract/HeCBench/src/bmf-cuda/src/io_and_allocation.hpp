#ifndef IO_AND_ALLOCATION
#define IO_AND_ALLOCATION

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath> // log2
#include <omp.h>

#include "config.h"
#include "helper/rngpu.hpp"

// safe division
#ifndef SDIV
#define SDIV(x,y)(((x)+(y)-1)/(y))
#endif

using std::string;
using std::vector;

float getInitChance(float density, uint8_t factorDim) {
  float threshold;

  switch(INITIALIZATIONMODE) {
    case 1:
      threshold = (sqrt(1 - pow(1 - density, float(1) / factorDim)));
      break;
    case 2:
      threshold = (density / 100);
      break;
    case 3:
      threshold = (density);
      break;
    default:
      threshold = 0;
      break;
  }
  return threshold;
}

void generate_random_matrix(const int height, const int width, const uint8_t factorDim, const int num_kiss, 
    vector<uint32_t> &Ab, vector<uint32_t> &Bb, vector<uint32_t> &C0b,
    float &density)
{
  uint32_t bit_vector_mask = uint32_t(~0) >> (32-factorDim);

  Ab.clear();
  Ab.resize(height, bit_vector_mask);
  Bb.clear();
  Bb.resize(width, bit_vector_mask);

  uint32_t seed = 42;
  fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

  for(int i=0; i < height; ++i) {
    // Ab[i] = bit_vector_mask;
    for(int kiss = 0; kiss < num_kiss; ++kiss)
      Ab[i] &= fast_kiss32(state);
  }
  for(int j=0; j < width; ++j) {
    // Bb[j] = bit_vector_mask;
    for(int kiss = 0; kiss < num_kiss; ++kiss)
      Bb[j] &= fast_kiss32(state);
  }

  // Malloc for C0b
  int padded_height_32 = SDIV(height, 32);
  int sizeCb = padded_height_32 * width;

  C0b.clear();
  C0b.resize(sizeCb, 0);

  // Create C
  int nonzeroelements = 0;

  for(int j=0; j < width; ++j) {
    for(int i=0; i < height; ++i) {
      if(Ab[i] & Bb[j]) {
        // int index = j*height+i;
        int vecId = i / 32 * width + j;
        int vecLane = i % 32;

        C0b[vecId] |= 1 << vecLane;

        ++nonzeroelements;
      }
    }
  }

  density = float(nonzeroelements) / height / width;

  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("MATRIX CREATION COMPLETE\n");
  printf("Height: %i\nWidth: %i\nNon-zero elements: %i\nDensity: %f\n",
      height, width, nonzeroelements, density);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

void generate_random_matrix(const int height, const int width, const uint8_t factorDim, const int num_kiss, 
    vector<float> &A, vector<float> &B, vector<uint32_t> &C0b,
    float &density)
{
  uint32_t bit_vector_mask = uint32_t(~0) >> (32-factorDim);

  A.clear();
  A.resize(height * factorDim, 0);
  B.clear();
  B.resize(width * factorDim, 0);

  uint32_t seed = 42;
  fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed);

  for(int i=0; i < height; ++i) {
    uint32_t mask = bit_vector_mask;
    for(int kiss = 0; kiss < num_kiss; ++kiss)
      mask &= fast_kiss32(state);
    for(int k=0; k < factorDim; ++k)
      A[i * factorDim + k] = (mask >> k) & 1 ? 1 : 0;
  }
  for(int j=0; j < width; ++j) {
    uint32_t mask = bit_vector_mask;
    for(int kiss = 0; kiss < num_kiss; ++kiss)
      mask &= fast_kiss32(state);
    for(int k=0; k < factorDim; ++k)
      B[j * factorDim + k] = (mask >> k) & 1 ? 1 : 0;
  }

  // float threshold = 1.0f;
  // for(int kiss = 0; kiss < num_kiss; ++kiss)
  //     threshold /= 2.0f;

  // for(int i=0; i < height * factorDim; ++i) {
  //     float random = (float) fast_kiss32(state) / UINT32_MAX;
  //     A[i] = random < threshold ? 1.0f : 0.0f;
  // }
  // for(int j=0; j < width * factorDim; ++j) {
  //     float random = (float) fast_kiss32(state) / UINT32_MAX;
  //     B[i] = random < threshold ? 1.0f : 0.0f;
  // }

  // Malloc for C0b
  size_t padded_height_32 = SDIV(height, 32);
  size_t sizeCb = padded_height_32 * width;

  C0b.clear();
  C0b.resize(sizeCb, 0);

  // Create C
  int nonzeroelements = 0;

  for(int j=0; j < width; ++j) {
    for(int i=0; i < height; ++i) {
      for (int k=0; k < factorDim; ++k) {
        if((A[i * factorDim + k] > 0.5f) && (B[j * factorDim + k] > 0.5f)) {
          int vecId = i / 32 * width + j;
          int vecLane = i % 32;
          C0b[vecId] |= 1 << vecLane;
          ++nonzeroelements;
          break;
        }
      }
    }
  }

  density = float(nonzeroelements) / height / width;

  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("MATRIX CREATION COMPLETE\n");
  printf("Height: %i\nWidth: %i\nNon-zero elements: %i\nDensity: %f\n",
      height, width, nonzeroelements, density);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

void readInputFileData(const string filename,
    vector<uint32_t> &C0b,
    int &height, int &width, 
    float &density)
{
  std::ifstream is {filename};

  if(!is.good()) throw std::runtime_error{"File " + filename +
    " could not be opened!"};

  std::uint64_t ones = 0;
  is >> height >> width >> ones;

  int padded_height_32 = SDIV(height, 32);
  int sizeCb = padded_height_32 * width;

  C0b.clear();
  C0b.resize(sizeCb,0);

  int nonzeroelements = 0;
  for(; ones > 0; --ones) {
    std::uint64_t r, c;
    is >> r >> c;
    int vecId = r / 32 * width + c;
    int vecLane = r % 32;
    C0b[vecId] |= 1 << vecLane;
    nonzeroelements++;
  }

  density = float(nonzeroelements) / height / width;

  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("READING OF DATA FILE COMPLETE\n");
  printf("Read height: %i\nRead width: %i\nNon-zero elements: %i\nDensity: %f\n",
      height, width, nonzeroelements, density);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
}

// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
bool endsWith(const string& s, const string& suffix) {
  return s.rfind(suffix) == (s.size()-suffix.size());
}


// Initialization of a factor, setting all bits of a row at once
void initFactorRowwise(vector<uint32_t> &Ab, 
    const int height,
    const uint8_t factorDim,
    const uint32_t seed, 
    const int randDepth)
{
  Ab.clear();

  if(randDepth < 16) {
    const uint32_t factorMask = UINT32_MAX >> (32-factorDim);
    Ab.resize(height, factorMask);

    // int counter = 0;
    #pragma omp parallel //reduce(+:counter)
    {
      fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + omp_get_thread_num());

      #pragma omp for
      for (int i = 0; i < height; i++) {
        for(int d = 0; d < randDepth; ++d) {
          Ab[i] &= fast_kiss32(state);
        }
        // if(Ab[i]) ++counter;
      }
    }
    // std::cout << "nonzero rows in factor: " << counter << std::endl;
  } else {
    Ab.resize(height, 0);
  }
}

// Initialization of a factor, setting every bits of a row on its own
void initFactorBitwise(vector<uint32_t> &Ab,
    const int height,
    const uint8_t factorDim,
    const uint32_t seed, 
    const uint32_t threshold_ui32)
{
  Ab.clear();
  Ab.resize(height, 0);

  // int counter = 0;
  #pragma omp parallel //reduce(+:counter)
  {
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + omp_get_thread_num());

    #pragma omp for
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < factorDim; j++) {
        if (fast_kiss32(state) < threshold_ui32)
          Ab[i] |= 1 << j;
      }
      // if(Ab[i]) ++counter;
    }
  }
  // std::cout << "nonzero rows in factor: " << counter << std::endl;
}

// Initialization of a factor, setting every bits of a row on its own
void initFactorBitwise( vector<float> &A,
    const int height,
    const uint8_t factorDim,
    const uint32_t seed,
    const uint32_t threshold_ui32)
{
  A.clear();
  A.resize(height * factorDim, 0);

  // int counter = 0;
  #pragma omp parallel //reduce(+:counter)
  {
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + omp_get_thread_num());

    #pragma omp for
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < factorDim; j++) {
        // two possibilities:
        // 1) set value 0 or 1
        // 2) set random value in [0,0.5] or [0.5,1]
        if(fast_kiss32(state) < threshold_ui32) {
          A[i * factorDim + j] = 1;
          // A[i * factorDim + j] = (fast_kiss32(state) / float(UINT32_MAX)) / 2 + 0.5f;
        } else {
          // A[i * factorDim + j] = (fast_kiss32(state) / float(UINT32_MAX)) / 2;
        }
      }
    }
  }
}

  template <typename factor_t>
void initFactor(vector<factor_t> &Ab,
    const int height,
    const uint8_t factorDim,
    const uint32_t seed,
    const float threshold)
{
  const int randDepth = -log2(threshold)+1;

  // std::cout << "Init threshold: " << threshold << std::endl;
  // std::cout << "Init rand depth: " << randDepth << " -> " << pow(2, -randDepth) << std::endl;

  if(randDepth < factorDim && std::is_same<factor_t, uint32_t>::value) {
    initFactorRowwise(Ab, height, factorDim, seed, randDepth);
  } else {
    initFactorBitwise(Ab, height, factorDim, seed, threshold * UINT32_MAX);
  }
}

// Write result factors to file
void writeFactorsToFiles(const string& filename,
    const vector<uint32_t>& Ab,
    const vector<uint32_t>& Bb,
    const uint8_t factorDim)
{
  using std::stringstream;
  using std::bitset;
  using std::ofstream;

  time_t now = time(0);
  // char* dt = ctime(&now);
  tm *ltm = localtime(&now);

  stringstream date;
  date << 1+ltm->tm_mon << '-' << ltm->tm_mday << '_' << ltm->tm_hour << ':' << ltm->tm_min << ':' << ltm->tm_sec;

  stringstream filename_A;
  filename_A << filename << "_factor_A_" << date.str() << ".data";
  stringstream filename_B;
  filename_B << filename << "_factor_B_" << date.rdbuf() << ".data";

  size_t height = Ab.size();

  int nonzeroelements = 0;
  for (unsigned int i = 0; i < height; i++){
    bitset<32> row(Ab[i]);
    nonzeroelements += row.count();
  }

  ofstream os_A(filename_A.str());
  if (os_A.good()){
    os_A << height << " " << int(factorDim) << " " << nonzeroelements << "\n";
    for (unsigned int i = 0; i < height; i++){
      // bitset<32> row(Ab[i] >> (32 - factorDim));
      // os_A << row << "\n";
      for(int k=0; k < factorDim; ++k)
        os_A << ((Ab[i] >> k) & 1 ? 1 : 0);
      os_A << "\n";
    }
    os_A.close();
  } else {
    std::cerr << "File " << filename_A.str() << " could not be openend!" << std::endl;
  }

  size_t width = Bb.size();

  nonzeroelements = 0;
  for (unsigned int j = 0; j < width; j++){
    bitset<32> col(Bb[j]);
    nonzeroelements += col.count();
  }

  ofstream os_B(filename_B.str());
  if(os_B.good()){
    os_B  << width << " " << int(factorDim) << " " << nonzeroelements << "\n";
    for (unsigned int j = 0; j < width; j++){
      // bitset<32> col(Bb[j] >> (32 - factorDim));
      // os_B << col << "\n";
      for(int k=0; k < factorDim; ++k)
        os_B << ((Bb[j] >> k) & 1 ? 1 : 0);
      os_B << "\n";
    }
    os_B.close();
  } else {
    std::cerr << "File " << filename_B.str() << " could not be openend!" << std::endl;
  }

  std::cout << "Writing to files \"" << filename_A.rdbuf() << "\" and \""
            << filename_B.rdbuf() << "\" complete" << std::endl;
}

  template<typename distance_t>
void writeDistancesToFile(const string& filename,
    const vector<distance_t>& distances)
{
  using std::stringstream;
  using std::bitset;
  using std::ofstream;

  time_t now = time(0);
  tm *ltm = localtime(&now);

  stringstream date;
  date << 1+ltm->tm_mon << '-' << ltm->tm_mday << '_' << ltm->tm_hour << ':' << ltm->tm_min << ':' << ltm->tm_sec;

  stringstream filename_d;
  filename_d << filename << "_distances_" << date.str() << ".txt";

  ofstream os(filename_d.str());
  if (os.good()){
    for (size_t i = 0; i < distances.size(); i++){
      if(i>0) os << "\n";
      os << distances[i];
    }
    os.close();
  } else {
    std::cerr << "File " << filename_d.str() << " could not be openend!" << std::endl;
  }

  std::cout << "Writing to files \"" << filename_d.rdbuf() << "\" complete" << std::endl;
}


#endif
