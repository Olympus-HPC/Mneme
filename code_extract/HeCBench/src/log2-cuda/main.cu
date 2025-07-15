#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "kernel.h"

int main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cout << "Usage: ./main <config filename>\n";
    return 1;
  }

  std::ifstream message_file (argv[1]);

  std::string placeholder;
  message_file >> placeholder;
  long ceilingVal;
  message_file >> ceilingVal;

  message_file >> placeholder;
  int repeat;
  message_file >> repeat;

  message_file >> placeholder;
  int precision_count;
  message_file >> precision_count;

  std::vector<int> precision(precision_count, 0);
  message_file >> placeholder;

  for (int i = 0; i < precision_count; ++i) {
    message_file >> precision[i];
  }

  std::vector<float> inputs;

  long i = 1;
  int increment = 1;

  while (i <= ceilingVal) {
    inputs.push_back((float) i);
    i += increment;
  }

  size_t inputs_size = inputs.size();

  std::cout << "Number of precision counts : " 
            << precision_count << std::endl
            << " Number of inputs to evaluate for each precision: "
            << inputs_size << std::endl
            << " Number of runs for each precision : " << repeat << std::endl;

  std::vector<float> empty_vector(inputs_size, 0);

#ifdef HOST
  // compute on the host
  std::vector<std::vector<float>> output_vals(precision_count, empty_vector);

  for(int i = 0; i < precision_count; ++i) {
    for(int k = 0; k < repeat; ++k) {
      for(size_t j = 0; j < inputs_size; ++j) {
        output_vals[i][j] = binary_log(inputs[j], precision[i]);
      }
    }
  }
#endif

  // store device results
  std::vector<float> d_output_vals(inputs_size * precision_count);

  // compute on the device
  log2_approx(inputs, d_output_vals, precision, 
              inputs.size(), precision_count, repeat);

  //generate references for MSE
  std::vector<float> ref_vals(inputs_size, 0);
  for (size_t i = 0; i < inputs_size; ++i)
    ref_vals[i] = log2f (inputs[i]);

  // compare references with host and device results
#ifdef HOST
  std::cout << "-------------- SUMMARY (Host results):" << " --------------" << std::endl<<std::endl;
  for (int i = 0; i < precision_count; ++i){
    std::cout << "----- Iterative approximation with " << precision[i] <<" bits of precision -----" << std::endl;
    float s = 0;
    for (size_t j = 0; j < inputs_size; ++j){
      s += (output_vals[i][j] - ref_vals[j]) * (output_vals[i][j] - ref_vals[j]);
    }
    s /= inputs.size();
    std::cout << "RMSE : " << sqrtf(s) << std::endl;
  }
#endif

  std::cout << "-------------- SUMMARY (Device results):" << " --------------" << std::endl<<std::endl;
  for (int i = 0; i < precision_count; ++i){
    std::cout << "----- Iterative approximation with " << precision[i] <<" bits of precision -----" << std::endl;
    float s = 0;
    for (size_t j = 0; j < inputs_size; ++j){
      s += (d_output_vals[i*inputs_size+j] - ref_vals[j]) * (d_output_vals[i*inputs_size+j] - ref_vals[j]);
    }
    s /= inputs.size();
    std::cout << "RMSE : " << sqrtf(s) << std::endl;
  }

  return 0;
}
