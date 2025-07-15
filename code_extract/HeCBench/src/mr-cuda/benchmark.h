#if !defined(_MSC_VER) || _MSC_VER >= 1800 // Visual Studio 2013 is the first version with inttypes.h
#include <inttypes.h>
#else
#define PRIu64 "llu"
#define PRIu32 "u"
#endif

#ifdef _MSC_VER
#define inline __inline
#endif

#if (defined(__CUDACC__) || defined (__HIPCC__))
#define ESS __host__  __device__
#else
#define ESS
#endif

#define BENCHMARK_ITERATIONS 100000

// With hashing only 3 bases are enough to test all numbers up to 2^64
// and 1 base to test all up to 2^32
// see http://probableprime.org/download/example-primality.c
#define BASES_CNT32 3
#define BASES_CNT_MAX BASES_CNT32

#include "myrand.h"
#include "mytime.h"

#include "sprp32.h"
#include "sprp32_sf.h"

// found by Gerhard Jaeschke
const uint32_t bases32[] = {2, 7, 61};
// see http://miller-rabin.appspot.com

#define SIZES_CNT32 4
#define SIZES_CNT_MAX SIZES_CNT32

static const char bits32[SIZES_CNT32] = {8,16,24,32};
static const uint32_t mask32[SIZES_CNT32] = {0xFFU,0xFFFFU,0xFFFFFFU,0xFFFFFFFFU};
static uint32_t n32[SIZES_CNT32][BENCHMARK_ITERATIONS];

#define WHEEL_PRODUCT 105
// wheel contains only odd numbers
static const unsigned char distancewheel[WHEEL_PRODUCT] = 
{0,8,6,4,2,0,0,2,0,0,2,0,4,2,0,0,4,2,0,2,0,0,2,0,4,2,0,4,2,0,0,4,2,0,2,
  0,0,4,2,0,2,0,4,2,0,6,4,2,0,2,0,0,2,0,0,2,0,6,4,2,0,4,2,0,2,0,4,2,0,0,
  2,0,4,2,0,0,4,2,0,4,2,0,2,0,0,2,0,4,2,0,0,4,2,0,2,0,0,2,0,0,8,6,4,2,0};
static const unsigned char wheeladvance[WHEEL_PRODUCT] = 
{10,0,0,0,0,2,4,0,2,4,0,6,0,0,2,6,0,0,4,0,2,4,0,6,0,0,6,0,0,2,6,0,0,4,0,
  2,6,0,0,4,0,6,0,0,8,0,0,0,4,0,2,4,0,2,4,0,8,0,0,0,6,0,0,4,0,6,0,0,2,4,
  0,6,0,0,2,6,0,0,6,0,0,4,0,2,4,0,6,0,0,2,6,0,0,4,0,2,4,0,2,10,0,0,0,0,2};

static void set_nprimes()
{
  myseed();
  for (int i = 0; i < SIZES_CNT32; i++)
    // simple PRIMEINC method - uniformity isn't important
    for (int j = 0; j < BENCHMARK_ITERATIONS; j++) {
      uint32_t n = (myrand32() & mask32[i]) | 1;
      n += distancewheel[(n >> 1) % WHEEL_PRODUCT];
      if (n < 5) n = 5;
      while (!efficient_mr32(bases32, 3, n))
        n += wheeladvance[(n >> 1) % WHEEL_PRODUCT];
      n32[i][j] = n;
    }
}

static void set_nintegers()
{
  myseed();
  for (int i = 0; i < SIZES_CNT32; i++)
    for (int j = 0; j < BENCHMARK_ITERATIONS; j++) {
      uint32_t n = (myrand32() & mask32[i]) | 1;
      if (n < 5) n = 5;
      n32[i][j] = n;
    }
}

void print_results(const char *bits_array, const int bits_limit, const int cnt_limit, uint64_t time_vals[][3][2])
{
  int i, j;

  printf("         ");
  for (i = 0; i < bits_limit; i++) {
    printf("|    %2d-bit integer   ", bits_array[i]);
  }
  printf("\n  bases  ");
  for (i = 0; i < bits_limit; i++) {
    printf("|  effcnt  |  simple  ");
  }
  printf("\n");

  for (i = 0; i < cnt_limit; i++) {
    const int cnt = i + 1;

    printf(" %d base%s", cnt, (cnt != 1 ? "s" : " "));

    for (j = 0; j < bits_limit; j++) {
      printf(" | %5" PRIu64 " ns", time_vals[j][i][0] / BENCHMARK_ITERATIONS);
      printf(" | %5" PRIu64 " ns", time_vals[j][i][1] / BENCHMARK_ITERATIONS);
    }
    printf("\n");
  }
  printf("\n");
}

