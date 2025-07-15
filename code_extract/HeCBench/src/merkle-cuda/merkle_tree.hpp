#pragma once
#include <cassert>
#include "rescue_prime.hpp"

// Given N -many leaves of Binary Merkle Tree computes all (N - 1) -many
// intermediate nodes by using Rescue Prime `merge` function, which
// merges two Rescue Prime digests into single of width 256 -bit
//
// N needs to be power of two
//
// Returns sum of all kernel execution times with nanosecond
// level granularity
//
// Have taken major motivation from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/c48b8555e07eb9557a20383cc9f3a4aeec834317/rescue_prime.c#L153-L164
// where I wrote similar routine using OpenCL
void
merklize_approach_1(const ulong* leaves,
                    ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const ulong4* mds,
                    const ulong4* ark1,
                    const ulong4* ark2);

