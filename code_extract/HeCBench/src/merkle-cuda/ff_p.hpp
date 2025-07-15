#pragma once

#include <cuda.h>

typedef unsigned long ulong;

// ulong4 is defined in vector_types.h included in NVHPC SDK

// Prime modulas of selected field, F_p,
// where p = 2 ** 64 - 2 ** 32 + 1
inline constexpr uint64_t MOD =
  ((((uint64_t)1 << 63) - ((uint64_t)1 << 31)) << 1) + 1;

// modular addition of two prime field elements
//
// note: operands doesn't necessarily need to ∈ F_p
// but second operand will be made `b % MOD`
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_add(uint64_t a, uint64_t b);

// modular subtraction of two prime field elements
//
// note: operands doesn't necessarily need to ∈ F_p
// but second operand will be made `b % MOD`
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_sub(uint64_t a, uint64_t b);

// modular mulitiplication of two prime field elements
//
// note: operands doesn't necessarily need to ∈ F_p
// but second operand will be made `b % MOD`
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_mult(uint64_t a, uint64_t b);

// modular exponentiation of prime field element by unsigned integer
//
// note: operands doesn't necessarily need to ∈ F_p
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_pow(uint64_t a, const uint64_t b);

// finds multiplicative inverse of field element, given that it's
// not additive identity
//
// note: if operand is not ∈ F_p, it's made so by performing
// modulo operation
//
// this function uses the fact a ** -1 = 1 / a = a ** (p - 2) ( mod p )
// where p = prime field modulas
//
// it raises operand to (p - 2)-th power, which is multiplicative
// inverse of operand
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_inv(uint64_t a);

// modular division of one prime field element by another one
//
// note: operands doesn't necessarily need to ∈ F_p
//
// it computes a * (b ** -1), uses already defined multiplicative
// inverse finder function
//
// return value may ∉ F_p, it's function invoker's
// responsibility to perform ret % MOD
__device__ uint64_t
ff_p_div(uint64_t a, uint64_t b);
