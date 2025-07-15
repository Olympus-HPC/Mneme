#include "rescue_prime.hpp"

inline __device__ ulong4 mul_hi (const ulong4 &a, const ulong4 &b)
{
  return { __umul64hi(a.x , b.x), __umul64hi(a.y , b.y), 
           __umul64hi(a.z , b.z), __umul64hi(a.w , b.w) };
}

inline __device__ ulong4 operator*(const ulong4 &a, const ulong4 &b)
{
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline __device__ ulong4 operator+(const ulong4 &a, const ulong4 &b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

inline __device__ ulong4 operator-(const ulong4 &a, const ulong4 &b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

inline __device__ ulong4 operator-(const ulong &a, const ulong4 &b)
{
  return {a - b.x, a - b.y, a - b.z, a - b.w};
}

inline __device__ ulong4 operator&(const ulong4 &a, const ulong b)
{
  return {a.x & b, a.y & b, a.z & b, a.w & b};
}

inline __device__ ulong4 operator>>(const ulong4 &a, const int b)
{
  return {a.x >> b, a.y >> b, a.z >> b, a.w >> b};
}

inline __device__ ulong4 operator<<(const ulong4 &a, const int b)
{
  return {a.x << b, a.y << b, a.z << b, a.w << b};
}

inline __device__ ulong4 operator<(const ulong4 &a, const ulong4 &b)
{
  return {(a.x < b.x) ? ULONG_MAX : 0, 
          (a.y < b.y) ? ULONG_MAX : 0,
          (a.z < b.z) ? ULONG_MAX : 0,
          (a.w < b.w) ? ULONG_MAX : 0};
}

inline __device__ ulong4 operator>(const ulong4 &a, const ulong4 &b)
{
  return {(a.x > b.x) ? ULONG_MAX : 0, 
          (a.y > b.y) ? ULONG_MAX : 0,
          (a.z > b.z) ? ULONG_MAX : 0,
          (a.w > b.w) ? ULONG_MAX : 0};
}

inline __device__ ulong4 operator>=(const ulong4 &a, const ulong4 &b)
{
  return {(a.x >= b.x) ? ULONG_MAX : 0, 
          (a.y >= b.y) ? ULONG_MAX : 0,
          (a.z >= b.z) ? ULONG_MAX : 0,
          (a.w >= b.w) ? ULONG_MAX : 0};
}

__device__
ulong4
ff_p_vec_mul_(ulong4 a, ulong4 b)
{
  ulong4 ab = a * b;
  ulong4 cd = mul_hi(a, b);
  ulong4 c = cd & MAX_UINT;
  ulong4 d = cd >> 32;

  ulong4 tmp_0 = ab - d;

  ulong4 und_0 = ab < d; // check if underflowed
  ulong4 tmp_2 = und_0 & MAX_UINT;

  ulong4 tmp_3 = tmp_0 - tmp_2;

  ulong4 tmp_4 = (c << 32) - c;

  ulong4 tmp_5 = tmp_3 + tmp_4;
  ulong4 ovr_0 = tmp_3 > (std::numeric_limits<uint64_t>::max() - tmp_4);
  ulong4 tmp_7 = ovr_0 & MAX_UINT;

  return tmp_5 + tmp_7;
}

__device__
void
ff_p_vec_mul(const ulong4* a,
             const ulong4* b,
             ulong4* const c)
{
  *(c + 0) = ff_p_vec_mul_(*(a + 0), *(b + 0));
  *(c + 1) = ff_p_vec_mul_(*(a + 1), *(b + 1));
  *(c + 2) = ff_p_vec_mul_(*(a + 2), *(b + 2));
}

__device__
ulong4
ff_p_vec_add_(const ulong4 &a, const ulong4 &b)
{
  // Following four lines are equivalent of writing
  // b % FIELD_MOD, which converts all lanes of `b` vector
  // into canonical representation
  const ulong4 mod_vec = make_ulong4(MOD, MOD, MOD, MOD);
  ulong4 over_0 = b >= mod_vec;
  ulong4 tmp_0 = (over_0 >> 63) * mod_vec;
  ulong4 b_ok = b - tmp_0;

  ulong4 tmp_1 = a + b_ok;
  ulong4 over_1 = a > (std::numeric_limits<uint64_t>::max() - b_ok);
  ulong4 tmp_2 = over_1 & MAX_UINT;

  ulong4 tmp_3 = tmp_1 + tmp_2;
  ulong4 over_2 = tmp_1 > (std::numeric_limits<uint64_t>::max() - tmp_2);
  ulong4 tmp_4 = over_2 & MAX_UINT;

  return tmp_3 + tmp_4;
}

__device__
void
ff_p_vec_add(const ulong4* a,
             const ulong4* b,
             ulong4* const c)
{
  *(c + 0) = ff_p_vec_add_(*(a + 0), *(b + 0));
  *(c + 1) = ff_p_vec_add_(*(a + 1), *(b + 1));
  *(c + 2) = ff_p_vec_add_(*(a + 2), *(b + 2));
}

__device__
void
apply_sbox(const ulong4* state_in, ulong4* const state_out)
{
  ulong4 state_in_2[3] = {};
  ff_p_vec_mul(state_in, state_in, state_in_2);

  ulong4 state_in_4[3] = {};
  ff_p_vec_mul(state_in_2, state_in_2, state_in_4);

  ulong4 state_in_6[3] = {};
  ff_p_vec_mul(state_in_2, state_in_4, state_in_6);

  ff_p_vec_mul(state_in, state_in_6, state_out);
}

__device__
void
apply_constants(const ulong4* state_in,
                const ulong4* cnst,
                ulong4* const state_out)
{
  ff_p_vec_add(state_in, cnst, state_out);
}

__device__
ulong
accumulate_vec4(ulong4 a)
{
  uint64_t v0 = ff_p_add(a.x, a.y);
  uint64_t v1 = ff_p_add(a.z, a.w);

  return static_cast<ulong>(ff_p_add(v0, v1));
}

__device__
ulong
accumulate_state(const ulong4* state)
{
  ulong v0 = accumulate_vec4(*(state + 0));
  ulong v1 = accumulate_vec4(*(state + 1));
  ulong v2 = accumulate_vec4(*(state + 2));

  return static_cast<ulong>(ff_p_add(v2, ff_p_add(v0, v1)));
}

__device__
void
apply_mds(const ulong4* state_in,
          const ulong4* mds,
          ulong4* const state_out)
{
  ulong4 scratch[3] = {};

  ff_p_vec_mul(state_in, mds + 0, scratch);
  ulong v0 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 3, scratch);
  ulong v1 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 6, scratch);
  ulong v2 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 9, scratch);
  ulong v3 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 12, scratch);
  ulong v4 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 15, scratch);
  ulong v5 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 18, scratch);
  ulong v6 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 21, scratch);
  ulong v7 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 24, scratch);
  ulong v8 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 27, scratch);
  ulong v9 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 30, scratch);
  ulong v10 = accumulate_state(scratch);

  ff_p_vec_mul(state_in, mds + 33, scratch);
  ulong v11 = accumulate_state(scratch);

  *(state_out + 0) = make_ulong4(v0, v1, v2, v3);
  *(state_out + 1) = make_ulong4(v4, v5, v6, v7);
  *(state_out + 2) = make_ulong4(v8, v9, v10, v11);
}

__device__
void
exp_acc(const ulong m,
        const ulong4* base,
        const ulong4* tail,
        ulong4* const out)
{
  ulong4 scratch[3] = {};

  *(out + 0) = *(base + 0);
  *(out + 1) = *(base + 1);
  *(out + 2) = *(base + 2);

  for (ulong i = 0; i < m; i++) {
    ff_p_vec_mul(out, out, scratch);

    *(out + 0) = *(scratch + 0);
    *(out + 1) = *(scratch + 1);
    *(out + 2) = *(scratch + 2);
  }

  ff_p_vec_mul(out, tail, scratch);

  *(out + 0) = *(scratch + 0);
  *(out + 1) = *(scratch + 1);
  *(out + 2) = *(scratch + 2);
}

__device__
void
apply_inv_sbox(const ulong4* state_in, ulong4* const state_out)
{
  ulong4 t1[3] = {};
  ff_p_vec_mul(state_in, state_in, t1);

  ulong4 t2[3] = {};
  ff_p_vec_mul(t1, t1, t2);

  ulong4 t3[3] = {};
  exp_acc(3, t2, t2, t3);

  ulong4 t4[3] = {};
  exp_acc(6, t3, t3, t4);

  ulong4 t5[3] = {};
  exp_acc(12, t4, t4, t5);

  ulong4 t6[3] = {};
  exp_acc(6, t5, t3, t6);

  ulong4 t7[3] = {};
  exp_acc(31, t6, t6, t7);

  ulong4 a[3] = {};
  ulong4 b[3] = {};
  ulong4 scratch[3] = {};

  ff_p_vec_mul(t7, t7, scratch);
  ff_p_vec_mul(t6, scratch, a);
  ff_p_vec_mul(a, a, scratch);
  ff_p_vec_mul(scratch, scratch, a);

  ff_p_vec_mul(t1, t2, scratch);
  ff_p_vec_mul(scratch, state_in, b);

  ff_p_vec_mul(a, b, state_out);
}

__device__
void
apply_permutation_round(const ulong4* state_in,
                        const ulong4* mds,
                        const ulong4* ark1,
                        const ulong4* ark2,
                        ulong4* const state_out)
{
  ulong4 scratch_0[3] = {};
  ulong4 scratch_1[3] = {};
  ulong4 scratch_2[3] = {};

  apply_sbox(state_in, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark1, scratch_2);

  apply_inv_sbox(scratch_2, scratch_0);
  apply_mds(scratch_0, mds, scratch_1);
  apply_constants(scratch_1, ark2, state_out);
}

__device__
void
apply_rescue_permutation(const ulong4* state_in,
                         const ulong4* mds,
                         const ulong4* ark1,
                         const ulong4* ark2,
                         ulong4* const state_out)
{
  ulong4 scratch_0[3] = {};
  ulong4 scratch_1[3] = {};
  ulong4 scratch_2[3] = {};

  apply_permutation_round(state_in, mds, ark1 + 0, ark2 + 0, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1 + 3, ark2 + 3, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1 + 6, ark2 + 6, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1 + 9, ark2 + 9, scratch_0);
  apply_permutation_round(scratch_0, mds, ark1 + 12, ark2 + 12, scratch_1);
  apply_permutation_round(scratch_1, mds, ark1 + 15, ark2 + 15, scratch_2);
  apply_permutation_round(scratch_2, mds, ark1 + 18, ark2 + 18, state_out);
}

__device__
void
hash_elements(const ulong* input_elements,
              const ulong count,
              ulong* const hash,
              const ulong4* mds,
              const ulong4* ark1,
              const ulong4* ark2)
{
  ulong4 state[3] = { make_ulong4(0, 0, 0, 0),
                      make_ulong4(0, 0, 0, 0),
                      make_ulong4(0, 0, 0, count % MOD) };
  ulong4 scratch[3] = {};

  ulong i = 0;
  for (ulong j = 0; j < count; j++) {
    switch (i) {
      case 0:
        state[0].x = ff_p_add(state[0].x, *(input_elements + j));
        break;
      case 1:
        state[0].y = ff_p_add(state[0].y, *(input_elements + j));
        break;
      case 2:
        state[0].z = ff_p_add(state[0].z, *(input_elements + j));
        break;
      case 3:
        state[0].w = ff_p_add(state[0].w, *(input_elements + j));
        break;
      case 4:
        state[1].x = ff_p_add(state[1].x, *(input_elements + j));
        break;
      case 5:
        state[1].y = ff_p_add(state[1].y, *(input_elements + j));
        break;
      case 6:
        state[1].z = ff_p_add(state[1].z, *(input_elements + j));
        break;
      case 7:
        state[1].w = ff_p_add(state[1].w, *(input_elements + j));
        break;
    }

    if ((++i) % RATE_WIDTH == 0) {
      apply_rescue_permutation(state, mds, ark1, ark2, scratch);
      i = 0;

      *(state + 0) = *(scratch + 0);
      *(state + 1) = *(scratch + 1);
      *(state + 2) = *(scratch + 2);
    }
  }

  if (i > 0) {
    apply_rescue_permutation(state, mds, ark1, ark2, scratch);

    *(state + 0) = *(scratch + 0);
    *(state + 1) = *(scratch + 1);
    *(state + 2) = *(scratch + 2);
  }

  *(hash + 0) = state[0].x;
  *(hash + 1) = state[0].y;
  *(hash + 2) = state[0].z;
  *(hash + 3) = state[0].w;
}

__device__
void
merge(const ulong* input_hashes,
      ulong* const merged_hash,
      const ulong4* mds,
      const ulong4* ark1,
      const ulong4* ark2)
{
  ulong4 state[3] = { make_ulong4(*(input_hashes + 0),
                                  *(input_hashes + 1),
                                  *(input_hashes + 2),
                                  *(input_hashes + 3)),
                      make_ulong4(*(input_hashes + 4),
                                  *(input_hashes + 5),
                                  *(input_hashes + 6),
                                  *(input_hashes + 7)),
                      make_ulong4(0, 0, 0, RATE_WIDTH) };
  ulong4 scratch[3] = {};

  apply_rescue_permutation(state, mds, ark1, ark2, scratch);

  *(merged_hash + 0) = scratch[0].x;
  *(merged_hash + 1) = scratch[0].y;
  *(merged_hash + 2) = scratch[0].z;
  *(merged_hash + 3) = scratch[0].w;
}

void
prepare_mds(ulong4* const mds)
{
  for (size_t i = 0; i < STATE_WIDTH * 3; i++) {
    *(mds + i) = make_ulong4(
      MDS[i * 4 + 0], MDS[i * 4 + 1], MDS[i * 4 + 2], MDS[i * 4 + 3]);
  }
}

void
prepare_ark1(ulong4* const ark1)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark1 + i) = make_ulong4(
      ARK1[i * 4 + 0], ARK1[i * 4 + 1], ARK1[i * 4 + 2], ARK1[i * 4 + 3]);
  }
}

void
prepare_ark2(ulong4* const ark2)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark2 + i) = make_ulong4(
      ARK2[i * 4 + 0], ARK2[i * 4 + 1], ARK2[i * 4 + 2], ARK2[i * 4 + 3]);
  }
}
