#pragma once

#include <limits>
#include <map>
#include <string>
#include "atomics.h"

enum ReductionType { SUM, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mul", MUL}, {"div", DIV}, {"min", MIN}, {"max", MAX}
};

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline __host__ __device__ void update(scalar_t &val,
                                                scalar_t new_val) {
    if (REDUCE == SUM)
      val += new_val;
    else if (REDUCE == MUL)
      val *= new_val;
    else if (REDUCE == DIV)
      val /= new_val;
    else if ((REDUCE == MIN && new_val < val) ||
             (REDUCE == MAX && new_val > val)) {
      val = new_val;
    }
  }

  static inline __device__ void atomic_write(scalar_t *address, scalar_t val) {
    if (REDUCE == SUM)
      atomAdd(address, val);
    else if (REDUCE == MUL)
      atomMul(address, val);
    else if (REDUCE == DIV)
      atomDiv(address, val);
    else if (REDUCE == MIN)
      atomMin(address, val);
    else if (REDUCE == MAX)
      atomMax(address, val);
  }
};
