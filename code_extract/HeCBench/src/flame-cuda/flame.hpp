/*
 * (c) 2009-2010 Christoph Schied <Christoph.Schied@uni-ulm.de>
 *
 * This file is part of flame.
 *
 * flame is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * flame is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with flame.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef  __FLAME_HPP__
#define  __FLAME_HPP__

#define WIDTH 800
#define HEIGHT 600
#define BPP 32

#define NUM_FUNCTIONS 20
#define VARIATIONS_PER_FUNCTION 5
#define NUM_VARIATIONS 49
#define NUM_THREADS (1 << 14)
#define THREADS_PER_BLOCK 32
#define NUM_POINTS_PER_THREAD 64
#define NUM_ITERATIONS 15
#define NUM_RANDOMS (1 << 22)
#define NUM_PERMUTATIONS 2048

struct VariationParameter {
  int idx;
  float factor;
};

struct ConstMemParams {
  VariationParameter variation_parameters[NUM_FUNCTIONS][VARIATIONS_PER_FUNCTION];
  float function_colors[NUM_FUNCTIONS];
  float pre_transform_params[NUM_FUNCTIONS][6];
  float post_transform_params[NUM_FUNCTIONS][6];
  int frame_counter;
  int enable_sierpinski;
  int thread_function_mapping[NUM_FUNCTIONS];
};

struct PermSortElement
{
  int value;
  unsigned short idx;

  int operator<(const struct PermSortElement &f) const {
    return this->value < f.value;
  }
};


unsigned mersenne_twister(unsigned *mersenne_state);

float radical_inverse(unsigned int n, unsigned int base);

void check_cuda_error(const char *msg);

extern int num_samples;
extern char *random_samples;
extern int frame_counter;
extern ConstMemParams const_mem_params;
extern float function_weights[NUM_FUNCTIONS];



#endif  /*__FLAME_HPP__*/
