#ifndef _IID_PERMUTATION_TESTING_H_
#define _IID_PERMUTATION_TESTING_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <mutex> // std::mutex

using namespace std;
#include "bzip/bzlib.h"

#define VERBOSE            0
#define NUMBER_OF_SAMPLES  1000000
#define BLOCK              10
#define THREAD             250
#define PARALLELISM        (BLOCK) * (THREAD)

// permutation_testing.cpp
bool permutation_testing(uint8_t *data, uint32_t size, uint32_t len, uint32_t numparallel, uint32_t numblock, uint32_t numthread, bool verbose);

// utils.cpp
uint32_t input_by_user(uint32_t *samplesize, uint32_t *len, uint32_t *numofparallel, uint32_t *numblock, uint32_t *numthread,
	               bool *verbose, const char *in_file_name);
int read_data_from_file(uint8_t *data, uint32_t size, uint32_t len, const char *in_file_name);
void print_original_test_statistics(double *results);
void print_counters(uint32_t *counts);
void seed(uint64_t *xoshiro256starstarState);
void xoshiro_jump(unsigned int jump_count, uint64_t *xoshiro256starstarState);
uint64_t randomRange64(uint64_t s, uint64_t *xoshiro256starstarState);
void FYshuffle(uint8_t data[], const int sample_size, uint64_t *xoshiro256starstarState);


// statistical_test.cpp
void calculate_statistics(double *dmean, double *dmedian, uint8_t *data, uint32_t size, uint32_t len);
int run_tests(double *results, double dmean, double dmedian, uint8_t *data, uint32_t size, uint32_t len);
void excursion_test(double *out, const double dmean, const uint8_t *data, const uint32_t len);
void directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data, const uint32_t len);
void runs_based_on_median(double *out_num, double *out_len, const double dmedian, const uint8_t *data, const uint32_t len);
int collision_test_statistic(double *out_avg, double *out_max, const uint8_t *data, const uint32_t size, const uint32_t len);
void periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len, uint32_t lag);
void compression(double *out, const uint8_t *data, const uint32_t len, const uint32_t size);

void conversion1(uint8_t *bdata, const uint8_t *data, const uint32_t len);
void conversion2(uint8_t *bdata, const uint8_t *data, const uint32_t len);

// gpu_permutation_testing.cu
bool gpu_permutation_testing(double *gpu_runtime, uint32_t *counts, double *results, double mean, double median,
                             uint8_t *data, uint32_t size, uint32_t len, uint32_t N, uint32_t num_block, uint32_t num_thread);

#endif // !IID_PERMUTATION_TESTING_H_
