#include "ff_p.hpp"
#include <climits>

__device__
uint64_t
ff_p_add(uint64_t a, uint64_t b)
{
  if (b >= MOD) {
    b -= MOD;
  }

  uint64_t res_0 = a + b;
  bool over_0 = a > UINT64_MAX - b;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));

  uint64_t res_1 = res_0 + tmp_0;
  bool over_1 = res_0 > UINT64_MAX - tmp_0;

  uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(over_1 ? 1 : 0));
  uint64_t res = res_1 + tmp_1;

  return res;
}

__device__
uint64_t
ff_p_sub(uint64_t a, uint64_t b)
{
  if (b >= MOD) {
    b -= MOD;
  }

  uint64_t res_0 = a - b;
  bool under_0 = a < b;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));

  uint64_t res_1 = res_0 - tmp_0;
  bool under_1 = res_0 < tmp_0;

  uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(under_1 ? 1 : 0));
  uint64_t res = res_1 + tmp_1;

  return res;
}

__device__
uint64_t
ff_p_mult(uint64_t a, uint64_t b)
{
  if (b >= MOD) {
    b -= MOD;
  }

  uint64_t ab = a * b;
  uint64_t cd = __umul64hi(a, b);
  uint64_t c = cd & 0x00000000ffffffff;
  uint64_t d = cd >> 32;

  uint64_t res_0 = ab - d;
  bool under_0 = ab < d;

  uint32_t zero = 0;
  uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));
  res_0 -= tmp_0;

  uint64_t tmp_1 = (c << 32) - c;

  uint64_t res_1 = res_0 + tmp_1;
  bool over_0 = res_0 > UINT64_MAX - tmp_1;

  uint64_t tmp_2 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));
  uint64_t res = res_1 + tmp_2;

  return res;
}

__device__
uint64_t
ff_p_pow(uint64_t a, const uint64_t b)
{
  if (b == 0) {
    return 1;
  }

  if (b == 1) {
    return a;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t r = b & 0b1 ? a : 1;
  for (uint8_t i = 1; i < 64 - __clzll(b); i++) {
    a = ff_p_mult(a, a);
    if ((b >> i) & 0b1) {
      r = ff_p_mult(r, a);
    }
  }
  return r;
}

__device__
uint64_t
ff_p_inv(uint64_t a)
{
  if (a >= MOD) {
    a -= MOD;
  }

  if (a == 0) {
    // ** no multiplicative inverse of additive identity **
    //
    // I'm not throwing an exception from here, because
    // this function is supposed to be invoked from
    // kernel body, where exception throwing is not (yet) allowed !
    return 0;
  }

  const uint64_t exp = MOD - 2;
  return ff_p_pow(a, exp);
}

__device__
uint64_t
ff_p_div(uint64_t a, uint64_t b)
{
  if (b == 0) {
    // ** no multiplicative inverse of additive identity **
    //
    // I'm not throwing an exception from here, because
    // this function is supposed to be invoked from
    // kernel body, where exception throwing is not (yet) allowed !
    return 0;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t b_inv = ff_p_inv(b);
  return ff_p_mult(a, b_inv);
}
