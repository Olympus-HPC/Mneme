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


#include "iteration.cu"

/* Each thread has a fixed assigned function and picks a random data-set by
 * indexing into the data array through a permutation array. The threads write
 * out their calculated data and read another randomly selected point by
 * indexing into another permutation array.
 *
 * This approach needs synchronization across all threads and threadblocks. To
 * achieve this, the algorithm has been split into three kernels:
 *  - initialize: initializes the iterations by selecting a random point of
 *    the set of starting points and applies one iteration
 *  - iterate: reads one point, iterates, writes out
 *  - generate points: reads iterated points and generates a bigger amount of
 *    points out of those points.
 *
 */

/* calculates the function a thread has to evaluate by using a binary search
 */
__device__ int
get_function_idx(int idx, const ConstMemParams &params)
{
  int func;
  int start = 0, end = NUM_FUNCTIONS;

  /* use warp size granularity, decreases branch divergence */
  idx &= ~31;

  for(int i = 0; i < 6; i++) {
    func = (start + end) / 2;
    if(params.thread_function_mapping[func] <= idx)
      start = func + 1;
    else
      end = func;
  }

  return func;
}

/* selects a random point out of the set of startingpoints and calculates one
 * iteration
 */
__global__ void
kernel_initialize(short2 *short_points, 
                  short *colors, 
                  const unsigned short *perms, 
                  const int perm_num,
                  float2 *start_pos,
                  const float *random_numbers,
                  const ConstMemParams params)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int function = get_function_idx(idx, params);


  int perm_idx = perms[NUM_THREADS * perm_num + idx];
  float2 point = start_pos[perm_idx];

  float color = 0.5f;

  iteration_fractal_flame(point, color, function, idx, random_numbers, params);

  colors[idx] = __float2half_rn(color);
  short_points[idx].x = __float2half_rn(point.x);
  short_points[idx].y = __float2half_rn(point.y);
}

/* reads one point, applies one iteration step and writes the point out */
__global__ void
kernel_iterate(short2 *short_points, 
               short *colors, 
               const unsigned short *perms,
               const int perm_num,
               const float *random_numbers,
               const ConstMemParams params)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int function = get_function_idx(idx, params);

  float2 point;
  float color;

  int perm_idx = perms[NUM_THREADS * perm_num + idx];

  color = __half2float(colors[perm_idx]);
  point.x = __half2float(short_points[perm_idx].x);
  point.y = __half2float(short_points[perm_idx].y);

  iteration_fractal_flame(point, color, function, idx, random_numbers, params);

  colors[perm_idx] = __float2half_rn(color);
  short_points[perm_idx].x = __float2half_rn(point.x);
  short_points[perm_idx].y = __float2half_rn(point.y);
}

/* after applying n iterations, the points converged close enough. new points
 * are created out of these converged points by applying another iteration
 * step and these new points get written into the destination VBO
 */
__global__ void
kernel_generate_points(float3 *vertices,
                       short2 *short_points, 
                       short *colors, 
                       const unsigned short *perms, 
                       const int perm_num,
                       const float *random_numbers,
                       const ConstMemParams params)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int function = get_function_idx(idx, params);

  float2 point;
  float color;

  for(int i = 0; i < NUM_POINTS_PER_THREAD; i++) {
    int perm_idx = perms[((perm_num + i) % NUM_PERMUTATIONS) *
      NUM_THREADS + idx];

    short2 _p = short_points[perm_idx];
    point.x = __half2float(_p.x);
    point.y = __half2float(_p.y);
    color = __half2float(colors[perm_idx]);

    iteration_fractal_flame(point, color, function, idx, random_numbers, params);

    vertices[idx + i * NUM_THREADS].x = point.x;
    vertices[idx + i * NUM_THREADS].y = point.y;
    vertices[idx + i * NUM_THREADS].z = color;
  }
}

