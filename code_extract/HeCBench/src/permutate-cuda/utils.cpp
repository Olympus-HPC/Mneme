#include "header.h"

#define SWAP(x, y) do { int s = x; x = y; y = s; } while(0)

uint32_t input_by_user(uint32_t *samplesize, uint32_t *len, uint32_t *numofparallel,
                       uint32_t *numblock, uint32_t *numthread, bool *verbose,
                       const char *in_file_name)
{
  FILE *fin;
  uint32_t user_len = 1000000;              /* the number of samples in the input data (= len) */
  uint32_t file_size = 1000000;             /* the size of the input file */
  uint32_t user_num_block = 10;           /* the number of GPU thread blocks */
  uint32_t user_num_thread = 250;           /* the number of threads using in the GPU kernel */
  uint32_t user_num_iteration_in_parallel;  /* the number of iterations processing in parallel on the GPU */
  uint32_t verbose_flag;                    /* optional verbosity flag for more output */

  if ((fin = fopen(in_file_name, "rb")) == NULL) {
    printf("File open fails. Please re-enter a file path\n");
    return 1;
  }

  // printf("\n[the_number_of_samples]: %u. It must be at least 1 million samples.\n", user_len);
  if (user_len != 0) {
    if (user_len < 1000000) {
      printf(" $the_number_of_samples must be at least 1,000,000. \n");
      return 1;
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    if (file_size < user_len) {
      printf("The size of the file(%d-byte) is smaller than [the_number_of_samples].\n", file_size);
      return 1;
    }
    *len = user_len;
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  if (file_size < 1000000) {
    printf("The size of the file(%d-byte) is smaller than the size required for testing(= 1,000,000).\n", file_size);
    fclose(fin);
    return 1;
  }
  fclose(fin);

  *samplesize = 8;
  // printf("\n[bits_per_symbol]: %u. It must be between 1-8, inclusive.\n", *samplesize);
  if ((*samplesize < 1) || (*samplesize > 8)) {
    printf("[bits_per_symbol] must be between 1-8.\n");
    return 1;
  }

  user_num_iteration_in_parallel = (user_num_block * user_num_thread);

  if (user_num_iteration_in_parallel != (user_num_block * user_num_thread)) {
    printf("[num_iteration_in_parallel] must be equal to [num_thread_block] x [num_thread_per_block].\n");
    return 1;
  }
  *numofparallel = user_num_iteration_in_parallel;
  *numblock = user_num_block;
  *numthread = user_num_thread;

  printf("\n[verbose] is off by default.\n");
  verbose_flag = 0;
  if (verbose_flag) *verbose = true;
  printf("\n");

  return 0;
}

int read_data_from_file(uint8_t *data, uint32_t size, uint32_t len, const char *in_file_name)
{
  FILE *fin;
  uint8_t temp = 0;
  uint8_t mask = (1 << size) - 1;
  uint32_t i = 0;

  char filename[200];
  sprintf(filename, "%s", in_file_name);
  if ((fin = fopen(in_file_name, "rb")) == NULL) {
    printf("File open fails. \n");
    return 1;
  }
  for (i = 0; i < len; i++) {
    temp = 0;
    fread(&temp, sizeof(unsigned char), 1, fin);
    data[i] = (temp&mask);
  }
  fclose(fin);

  return 0;
}

void print_original_test_statistics(double *results)
{
  printf(">---- Origianl test statistics: \n");
  printf("                        Excursion test = %0.4f \n", results[0]);
  printf("            Number of directional runs = %0.0f \n", results[1]);
  printf("            Length of directional runs = %0.0f \n", results[2]);
  printf("    Numbers of increases and decreases = %0.0f \n", results[3]);
  printf("        Number of runs based on median = %0.0f \n", results[4]);
  printf("        Length of runs based on median = %0.0f \n", results[5]);
  printf("      Average collision test statistic = %0.4f \n", results[6]);
  printf("      Maximum collision test statistic = %0.0f \n", results[7]);
  printf("           Periodicity test (lag =  1) = %0.0f \n", results[8]);
  printf("           Periodicity test (lag =  2) = %0.0f \n", results[9]);
  printf("           Periodicity test (lag =  8) = %0.0f \n", results[10]);
  printf("           Periodicity test (lag = 16) = %0.0f \n", results[11]);
  printf("           Periodicity test (lag = 32) = %0.0f \n", results[12]);
  printf("            Covariance test (lag =  1) = %0.0f \n", results[13]);
  printf("            Covariance test (lag =  2) = %0.0f \n", results[14]);
  printf("            Covariance test (lag =  8) = %0.0f \n", results[15]);
  printf("            Covariance test (lag = 16) = %0.0f \n", results[16]);
  printf("            Covariance test (lag = 32) = %0.0f \n", results[17]);
  printf("                      Compression test = %0.0f \n", results[18]);
  printf(">------------------------------------------------------< \n\n");
}

void print_counters(uint32_t *counts)
{
  printf(">---- The ranking(count) of the original test statistics: \n");
  printf("  #Test   Counter 1  Counter 2  Counter 3\n");
  for (uint32_t i = 0; i < 19; i++)
    printf("    %2d  %7d   %7d   %7d \n", i + 1, counts[3 * i], counts[3 * i + 1], counts[3 * i + 2]);
  printf(">------------------------------------------------------< \n\n");
}

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256starstar(uint64_t *xoshiro256starstarState)
{
  const uint64_t result_starstar = rotl(xoshiro256starstarState[1] * 5, 7) * 9;
  const uint64_t t = xoshiro256starstarState[1] << 17;

  xoshiro256starstarState[2] ^= xoshiro256starstarState[0];
  xoshiro256starstarState[3] ^= xoshiro256starstarState[1];
  xoshiro256starstarState[1] ^= xoshiro256starstarState[2];
  xoshiro256starstarState[0] ^= xoshiro256starstarState[3];

  xoshiro256starstarState[2] ^= t;

  xoshiro256starstarState[3] = rotl(xoshiro256starstarState[3], 45);

  return result_starstar;
}

void seed(uint64_t *xoshiro256starstarState) {

  srand(123);
  for (int i = 0; i < 4; i++) {
    xoshiro256starstarState[i] = rand();
    xoshiro256starstarState[i] = (uint64_t)(xoshiro256starstarState[i] << 31);
    xoshiro256starstarState[i] ^= (uint64_t)(rand() & 0xFFFFFFFF);
  }
}

void xoshiro_jump(unsigned int jump_count, uint64_t *xoshiro256starstarState) {
  static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;

  for (unsigned int j = 0; j < jump_count; j++) {
    for (unsigned int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
      for (unsigned int b = 0; b < 64; b++) {
        if (JUMP[i] & ((uint64_t)1) << b) {
          s0 ^= xoshiro256starstarState[0];
          s1 ^= xoshiro256starstarState[1];
          s2 ^= xoshiro256starstarState[2];
          s3 ^= xoshiro256starstarState[3];
        }
        xoshiro256starstar(xoshiro256starstarState);
      }

    xoshiro256starstarState[0] = s0;
    xoshiro256starstarState[1] = s1;
    xoshiro256starstarState[2] = s2;
    xoshiro256starstarState[3] = s3;
  }
}

uint64_t randomRange64(uint64_t s, uint64_t *xoshiro256starstarState) {
  uint64_t x;
  uint64_t m[2];
  uint64_t l;
  unsigned int R0 = 0, R1 = 0, R2 = 0;
  unsigned long UV;
  unsigned int out[4] = { 0, };
  unsigned int BigInt1[2], BigInt2[2];
  unsigned int pre_eps, eps;
  int k, i, j;

  x = xoshiro256starstar(xoshiro256starstarState);

  if (UINT64_MAX == s) {
    return x;
  }
  else {
    s++; // We want an integer in the range [0,s], not [0,s)

    //m[1] = (__uint128_t)x * (__uint128_t)s;
    BigInt1[0] = x & 0xFFFFFFFF; BigInt1[1] = (x >> 32) & 0xFFFFFFFF;
    BigInt2[0] = s & 0xFFFFFFFF; BigInt2[1] = (s >> 32) & 0xFFFFFFFF;
    R0 = 0; R1 = 0; R2 = 0;
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    for (k = 0; k < 3; k++) {
      for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
          if (i + j == k) {
            UV = (unsigned long)BigInt1[i] * (unsigned long)BigInt2[j];
            R0 = R0 + (unsigned int)(UV & 0xFFFFFFFF);
            if (R0 < (unsigned int)(UV & 0xFFFFFFFF)) pre_eps = 1;
            else pre_eps = 0;

            R1 = R1 + (unsigned int)((UV >> 32) & 0xFFFFFFFF);
            if (R1 < (unsigned int)((UV >> 32) & 0xFFFFFFFF)) eps = 1;
            else eps = 0;

            R1 = R1 + pre_eps;
            if (R1 < pre_eps) eps++;
            pre_eps = eps;
            R2 = R2 + pre_eps;
          }
        }
      }
      out[k] = R0;
      R0 = R1;
      R1 = R2;
      R2 = 0;
    }
    out[3] = R0;

    m[0] = (((uint64_t)out[3] << 32) & 0xFFFFFFFF00000000) ^ ((uint64_t)out[2] & 0xFFFFFFFF);
    m[1] = (((uint64_t)out[1] << 32) & 0xFFFFFFFF00000000) ^ ((uint64_t)out[1] & 0xFFFFFFFF);

    l = (uint64_t)m[1]; //This is m mod 2^64

    if (l < s) {
      uint64_t t = (0x100000000 - (uint64_t)(s)) % s; //t = (2^64 - s) mod s (by definition of unsigned arithmetic in C)
      while (l < t) {
        x = xoshiro256starstar(xoshiro256starstarState);
        //m = (__uint128_t)x * (__uint128_t)s;
        BigInt1[0] = x & 0xFFFFFFFF; BigInt1[1] = (x >> 32) & 0xFFFFFFFF;
        BigInt2[0] = s & 0xFFFFFFFF; BigInt2[1] = (s >> 32) & 0xFFFFFFFF;
        R0 = 0; R1 = 0; R2 = 0;
        out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
        for (k = 0; k < 3; k++) {
          for (i = 0; i < 2; i++) {
            for (j = 0; j < 2; j++) {
              if (i + j == k) {
                UV = (unsigned long)BigInt1[i] * (unsigned long)BigInt2[j];
                R0 = R0 + (unsigned int)(UV & 0xFFFFFFFF);
                if (R0 < (unsigned int)(UV & 0xFFFFFFFF)) pre_eps = 1;
                else pre_eps = 0;

                R1 = R1 + (unsigned int)((UV >> 32) & 0xFFFFFFFF);
                if (R1 < (unsigned int)((UV >> 32) & 0xFFFFFFFF)) eps = 1;
                else eps = 0;

                R1 = R1 + pre_eps;
                if (R1 < pre_eps) eps++;
                pre_eps = eps;
                R2 = R2 + pre_eps;
              }
            }
          }
          out[k] = R0;
          R0 = R1;
          R1 = R2;
          R2 = 0;
        }
        out[3] = R0;

        m[0] = (((uint64_t)out[3] << 32) & 0xFFFFFFFF00000000) ^ ((uint64_t)out[2] & 0xFFFFFFFF);
        m[1] = (((uint64_t)out[1] << 32) & 0xFFFFFFFF00000000) ^ ((uint64_t)out[1] & 0xFFFFFFFF);
        l = (uint64_t)m[1]; //This is m mod 2^64
      }
    }

    return (uint64_t)m[0]; //return floor(m/2^64)
  }
}

void FYshuffle(uint8_t data[], const int sample_size, uint64_t *xoshiro256starstarState) {
  long int r;
  static mutex shuffle_mutex;
  unique_lock<mutex> lock(shuffle_mutex);

  for (long int i = sample_size - 1; i > 0; --i) {
    r = (long int)randomRange64((uint64_t)i, xoshiro256starstarState);
    SWAP(data[r], data[i]);
  }
}
