#ifndef RNGPU_HPP
#define RNGPU_HPP

#include <stdint.h>

#if defined(__CUDACC__)
#define INLINE_QUALIFIERS __host__ __device__ __forceinline__
#else
#include <math.h>
#define INLINE_QUALIFIERS
#endif

//////////////////////////////////////////////////////////////////////////////
// Thomas Mueller hash for initialization of rngs
// http://stackoverflow.com/questions/664014/
//        what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
//////////////////////////////////////////////////////////////////////////////

#define hash32(x) (nvidia_hash(x))

INLINE_QUALIFIERS
uint32_t mueller_hash(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

INLINE_QUALIFIERS
uint32_t nvidia_hash(uint32_t x) {

    x = (x + 0x7ed55d16) + (x << 12);
    x = (x ^ 0xc761c23c) ^ (x >> 19);
    x = (x + 0x165667b1) + (x <<  5);
    x = (x + 0xd3a2646c) ^ (x <<  9);
    x = (x + 0xfd7046c5) + (x <<  3);
    x = (x ^ 0xb55a4f09) ^ (x >> 16);

    return x;
}

//////////////////////////////////////////////////////////////////////////////
// hash random number generator (period unknown)
//////////////////////////////////////////////////////////////////////////////

struct hashrng_state32_t {
    uint32_t x;
};

template <uint32_t num_iters=4> INLINE_QUALIFIERS
hashrng_state32_t get_initial_hashrng_state32(uint32_t seed) {

    hashrng_state32_t state;

    state.x = !seed ? 4294967295 : seed;
    for (uint32_t iter = 0; iter < num_iters; iter++)
        state.x = hash32(state.x);

    return state;
}

//////////////////////////////////////////////////////////////////////////////
// linear congruential generator
//////////////////////////////////////////////////////////////////////////////

struct lcg_state32_t{
    uint32_t x;
};

INLINE_QUALIFIERS
uint32_t lcg32(lcg_state32_t &state) {
    state.x = 314527869*state.x+1234567;
    return state.x;
}

template <uint32_t num_iters=8> INLINE_QUALIFIERS
lcg_state32_t get_initial_lcg_state32(uint32_t seed) {

    lcg_state32_t state;

    state.x = !seed ? 4294967295 : seed;
    for (uint32_t iter = 0; iter < num_iters; iter++)
        state.x = hash32(state.x);

    return state;
}

//////////////////////////////////////////////////////////////////////////////
// keep it simple stupid generator
//////////////////////////////////////////////////////////////////////////////

struct kiss_state32_t{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

INLINE_QUALIFIERS
uint32_t kiss32(kiss_state32_t &state) {

    // lcg
    state.x = 314527869*state.x+1234567;

    // xorshift
    state.y ^= state.y << 5;
    state.y ^= state.y >> 7;
    state.y ^= state.y << 22;

    // carry and multiply
    uint64_t t = 4294584393ULL*state.z+state.w;
    state.w = t;
    state.w = t >> 32;

    // combine
    return state.x+state.y+state.z;
}

template <uint32_t num_iters=8> INLINE_QUALIFIERS
kiss_state32_t get_initial_kiss_state32(const uint32_t seed) {

    kiss_state32_t state;

    state.w = !seed ? 4294967295 : seed;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        state.x = hash32(state.w);
        state.y = hash32(state.x);
        state.z = hash32(state.y);
        state.w = hash32(state.z);
    }

    return state;
}

//////////////////////////////////////////////////////////////////////////////
// fast keep it simple stupid generator
//////////////////////////////////////////////////////////////////////////////

struct fast_kiss_state32_t{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
    uint32_t c;
};

INLINE_QUALIFIERS
uint32_t fast_kiss32(fast_kiss_state32_t &state) {

    state.y ^= state.y << 5;
    state.y ^= state.y >> 7;
    state.y ^= state.y << 22;

    int32_t t = state.z+state.w+state.c;
    state.z  = state.w;
    state.c  = t < 0;
    state.w  = t & 2147483647;
    state.x += 1411392427;

    // combine
    return state.x+state.y+state.w;
}

template <uint32_t num_iters=8> INLINE_QUALIFIERS
fast_kiss_state32_t get_initial_fast_kiss_state32(const uint32_t seed) {

    fast_kiss_state32_t state;

    state.w = !seed ? 4294967295 : seed;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        state.x = hash32(state.w);
        state.y = hash32(state.x);
        state.z = hash32(state.y);
        state.w = hash32(state.z);
    }

    state.c = 0;

    return state;
}


//////////////////////////////////////////////////////////////////////////////
// xorwow random number generator
//////////////////////////////////////////////////////////////////////////////

struct xorwow_state32_t{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
    uint32_t v;
    uint32_t d;
};

INLINE_QUALIFIERS
uint32_t xorwow32(xorwow_state32_t &state) {
    
    uint32_t t = (state.x^(state.x>>2));
    state.x = state.y;
    state.y = state.z;
    state.z = state.w;
    state.v = (state.v^(state.v<<4))^(t^(t<<1));
    state.d = state.d+362437;

    return state.v+state.d;
}

template <uint32_t num_iters=8> INLINE_QUALIFIERS
xorwow_state32_t get_initial_xorwow_state32(uint32_t seed) {

    xorwow_state32_t state;

    state.d= !seed ? 4294967295 : seed;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        state.x = hash32(state.d);
        state.y = hash32(state.x);
        state.z = hash32(state.y);
        state.w = hash32(state.z);
        state.v = hash32(state.w);
        state.d = hash32(state.v);
    }

    return state;
}

//////////////////////////////////////////////////////////////////////////////
// meta functions for uniform sampling in [0, 1)
//////////////////////////////////////////////////////////////////////////////

template <class rng_t,
          class state_t> INLINE_QUALIFIERS
float uniform_float(rng_t rng, state_t * state) {

    uint32_t a = rng(state) >> 9;
    float x; *((unsigned int *)&x) = a | 0x3F800000;

    return x-1.0f;
}

template <class rng_t,
          class state_t> INLINE_QUALIFIERS
double uniform_double(rng_t rng, state_t * state) {

    uint32_t a = rng(state) >> 6;
    uint32_t b = rng(state) >> 5;
    double x = (a * 134217728.0 + b) / 9007199254740992.0;

    return x;
}

//////////////////////////////////////////////////////////////////////////////
// shuffle functions for arrays
//////////////////////////////////////////////////////////////////////////////

template <class rng_t,
          class state_t,
          class value_t,
          class index_t> INLINE_QUALIFIERS
void fisher_yates_shuffle(rng_t rng,
                          state_t * state,
                          value_t * array,
                          index_t length) {

    for (index_t i = length-1; i >= 1; i--) {
        index_t r = (double) rng(state) * (i + 1) / 4294967296.0;
        value_t temp = array[i];
        array[i] = array[r];
        array[r] = temp;
    }
}

#endif
