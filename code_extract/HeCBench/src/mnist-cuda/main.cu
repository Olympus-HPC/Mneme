#include <cstdio>
#include <chrono>
#include <cmath>
#include <cuda.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "kernels.h"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn(int iter);
static unsigned int classify(double data[28][28]);
static void test();
void forward_pass(double data[28][28]);
void back_pass();

static inline int loaddata()
{
  int s1 = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
                      &train_set, &train_cnt);
  int s2 = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
                      &test_set, &test_cnt);
  return s1 | s2;
}

// replace cublas function in the case n = 10
void snrm2(const int n, const float *x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));
  cudaMemcpy(r, x, n * sizeof(float), cudaMemcpyDeviceToHost);
  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

int main(int argc, const  char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iter = atoi(argv[1]);
  srand(123);
  if (loaddata() != 0) return 1;

  auto t1 = std::chrono::high_resolution_clock::now();
  learn(iter);
  test();
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Total time (learn + test) %lf secs \n", total_time / 1.0e6);
  return 0;
}

// Forward propagation of a single row in dataset
void forward_pass(double data[28][28])
{
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  float input[28][28];
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }

  l_input.setOutput((float *)input);

  fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
  apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

  fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
  apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

  fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
}

// Back propagation to update weights
void back_pass()
{
  bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

  bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

  bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);


  apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
}

static void learn(int iter)
{
  float err;

  fprintf(stdout ,"Learning\n");

  while (iter < 0 || iter-- > 0) {
    err = 0.0f;

    for (unsigned int i = 0; i < train_cnt; ++i) {
      float tmp_err;

      forward_pass(train_set[i].data);

      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      // Euclid distance of train_set[i]
      makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
      snrm2(10, l_f.d_preact, tmp_err);
      err += tmp_err;

      back_pass();
    }

    err /= train_cnt;
    fprintf(stdout, "error: %e\n", err);

    if (err < threshold) {
      fprintf(stdout, "Training complete, error less than threshold\n\n");
      break;
    }
  }
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
  forward_pass(data);

  float res[10];
  unsigned int max = 0;

  cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test()
{
  fprintf(stdout ,"Testing\n");

  int error = 0;

  for (unsigned int i = 0; i < test_cnt; ++i) {
    if (classify(test_set[i].data) != test_set[i].label) {
      ++error;
    }
  }

  fprintf(stdout, "Error Rate: %.2lf%%\n",
      double(error) / double(test_cnt) * 100.0);
}
