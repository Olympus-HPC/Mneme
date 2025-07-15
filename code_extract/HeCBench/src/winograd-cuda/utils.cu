#include "utils.h"

// F(2x2,3x3)

void WinogradConv2D_2x2_filter_transformation(DATA_TYPE* transformed_filter) {
  DATA_TYPE filter[3][3];

  filter[0][0] = +0.2;
  filter[1][0] = +0.5;
  filter[2][0] = -0.8;
  filter[0][1] = -0.3;
  filter[1][1] = +0.6;
  filter[2][1] = -0.9;
  filter[0][2] = +0.4;
  filter[1][2] = +0.7;
  filter[2][2] = +0.10;

  // filter transformation

  DATA_TYPE tmp_filter[4][3];

  // const float G[4][3] = {
  //     {1.0f, 0.0f, 0.0f},
  //     {0.5f, 0.5f, 0.5f},
  //     {0.5f, -0.5f, 0.5f},
  //     {0.0f, 0.0f, 1.0f}
  // };

  // G * g
  for (int j = 0; j < 3; j++) {
    tmp_filter[0][j] = filter[0][j];
    tmp_filter[1][j] = 0.5f * filter[0][j] + 0.5f * filter[1][j] + 0.5f * filter[2][j];
    tmp_filter[2][j] = 0.5f * filter[0][j] - 0.5f * filter[1][j] + 0.5f * filter[2][j];
    tmp_filter[3][j] = filter[2][j];
  }
  // g * Gt
  for (int i = 0; i < 4; i++) {
    transformed_filter[i * 4 + 0] = tmp_filter[i][0];
    transformed_filter[i * 4 + 1] = 0.5f * tmp_filter[i][0] + 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
    transformed_filter[i * 4 + 2] = 0.5f * tmp_filter[i][0] - 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
    transformed_filter[i * 4 + 3] = tmp_filter[i][2];
  }
}

void WinogradConv2D_2x2_omp(DATA_TYPE* input, DATA_TYPE* output, DATA_TYPE* transformed_filter, size_t* cpu_global_size) {
  // DATA_TYPE trasformed_filter[4][4];
  // WinogradConv2D_2x2_filter_transformation(trasformed_filter);

  int out_map_size = MAP_SIZE - 2;
  int tile_n = (out_map_size + 1) / 2;

  #pragma omp parallel
  for (int tile_i = 0; tile_i < cpu_global_size[0]; tile_i++) {
    #pragma omp for
    for (int tile_j = 0; tile_j < tile_n; tile_j++) {
      // input transformation

      DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= MAP_SIZE || y >= MAP_SIZE) {
            input_tile[i][j] = 0;
            continue;
          }
          input_tile[i][j] = input[x * MAP_SIZE + y];
        }
      }

      // const float Bt[4][4] = {
      //     {1.0f, 0.0f, -1.0f, 0.0f},
      //     {0.0f, 1.0f, 1.0f, 0.0f},
      //     {0.0f, -1.0f, 1.0f, 0.0f},
      //     {0.0f, 1.0f, 0.0f, -1.0f}
      // }

      // Bt * d
      // #pragma omp simd
      for (int j = 0; j < 4; j++) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
      }
      // d * B
      // #pragma omp simd
      for (int i = 0; i < 4; i++) {
        transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
        transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
      }

      // element-wise multiplication

      DATA_TYPE multiplied_tile[4][4];
      for (int i = 0; i < 4; i++) {
        // #pragma omp simd
        for (int j = 0; j < 4; j++) {
          multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
      }

      // output transformation

      DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

      // const float At[2][4] {
      //     {1.0f, 1.0f, 1.0f, 0.0f},
      //     {0.0f, 1.0f, -1.0f, -1.0f}
      // }

      // At * I
      // #pragma omp simd
      for (int j = 0; j < 4; j++) {
        tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
        tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
      }
      // I * A
      // #pragma omp simd
      for (int i = 0; i < 2; i++) {
        final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
        final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= out_map_size || y >= out_map_size) {
            continue;
          }
          output[x * out_map_size + y] = final_tile[i][j];
        }
      }

    }  // for tile_i
  }      // for tile_j
}

bool compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu) {
  int i, j, fail;
  fail = 0;

  // Compare a and b
  for (i = 0; i < (MAP_SIZE - 2); i++) {
    for (j = 0; j < (MAP_SIZE - 2); j++) {
      if (percentDiff(B[i * (MAP_SIZE - 2) + j], B_outputFromGpu[i * (MAP_SIZE - 2) + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
        fail++;
      }
    }
  }

  // Print results
  #ifdef VERBOSE
  printf("Error Threshold of %4.2f Percent: %d\n\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
  #endif
  return (fail == 0) ? true : false;
}

void WinogradConv2D_2x2(DATA_TYPE* input, DATA_TYPE* output, DATA_TYPE* transformed_filter) {

  int out_map_size = MAP_SIZE - 2;
  int tile_n = (out_map_size + 1) / 2;

  for (int tile_i = 0; tile_i < tile_n; tile_i++) {
    for (int tile_j = 0; tile_j < tile_n; tile_j++) {
      // input transformation

      DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= MAP_SIZE || y >= MAP_SIZE) {
            input_tile[i][j] = 0;
            continue;
          }
          input_tile[i][j] = input[x * MAP_SIZE + y];
        }
      }

      // const float Bt[4][4] = {
      //     {1.0f, 0.0f, -1.0f, 0.0f},
      //     {0.0f, 1.0f, 1.0f, 0.0f},
      //     {0.0f, -1.0f, 1.0f, 0.0f},
      //     {0.0f, 1.0f, 0.0f, -1.0f}
      // }

      // Bt * d
      for (int j = 0; j < 4; j++) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
      }
      // d * B
      for (int i = 0; i < 4; i++) {
        transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
        transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
      }

      // element-wise multiplication

      DATA_TYPE multiplied_tile[4][4];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
      }

      // output transformation

      DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

      // const float At[2][4] {
      //     {1.0f, 1.0f, 1.0f, 0.0f},
      //     {0.0f, 1.0f, -1.0f, -1.0f}
      // }

      // At * I
      for (int j = 0; j < 4; j++) {
        tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
        tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
      }
      // I * A
      for (int i = 0; i < 2; i++) {
        final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
        final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= out_map_size || y >= out_map_size) {
            continue;
          }
          output[x * out_map_size + y] = final_tile[i][j];
        }
      }

    }  // for tile_i
  }      // for tile_j
}


double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


float absVal(float a)
{
  if(a < 0)
  {
    return (a * -1);
  }
  else
  { 
    return a;
  }
}



float percentDiff(double val1, double val2)
{
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
  {
    return 0.0f;
  }

  else
  {
    return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
} 
