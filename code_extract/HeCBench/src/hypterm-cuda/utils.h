#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#define TOLERANCE 1e-3

#define D 308

template<typename T>
static T get_random() {
  return ((T)(rand())/(T)(RAND_MAX-1));
}

template<typename T>
static T* getRandom3DArray(int height, int width_y, int width_x) {
  T (*a)[D][D] = (T (*)[D][D])new T[height*D*D];
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width_y; j++)
      for (int k = 0; k < width_x; k++) {
        a[i][j][k] = get_random<T>() + 0.02121;
      }
  return (T*)a;
}

template<typename T>
static T* getZero3DArray(int height, int width_y, int width_x) {
  T (*a)[D][D] = (T (*)[D][D])new T[height*D*D];
  memset((void*)a, 0, sizeof(T) * height * width_y * width_x);
  return (T*)a;
}

template<typename T>
static double checkError3D
(int width_y, int width_x, const T *l_output, const T *l_reference, int z_lb,
 int z_ub, int y_lb, int y_ub, int x_lb, int x_ub) {
  const T (*output)[D][D] = (const T (*)[D][D])(l_output);
  const T (*reference)[D][D] = (const T (*)[D][D])(l_reference);
  double error = 0.0;
  double max_error = TOLERANCE, sum = 0.0;
  for (int i = z_lb; i < z_ub; i++) {
    for (int j = y_lb; j < y_ub; j++) {
      for (int k = x_lb; k < x_ub; k++) {
        sum += output[i][j][k];
        //printf ("real var1[%d][%d][%d] = %.6f and %.6f\n", i, j, k, reference[i][j][k], output[i][j][k]);
        double curr_error = fabs(output[i][j][k] - reference[i][j][k]);
        error += curr_error * curr_error;
        if (curr_error > max_error) {
          printf ("Values at index (%d,%d,%d) differ : %.6f and %.6f\n", i, j, k, reference[i][j][k], output[i][j][k]);
        }
      }
    }
  }
  printf ("checksum = %e\n", sum);
  return sqrt(error / ( (z_ub - z_lb) * (y_ub - y_lb) * (x_ub - x_lb)));
}

#endif
