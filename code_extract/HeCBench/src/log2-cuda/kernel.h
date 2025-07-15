#include <chrono>
#include <cuda.h>

typedef union type_caster_union { /* a union between a float and an integer */
  public:
    float f;
    uint32_t i;
} placeholder_name;

__host__ __device__
float binary_log(float input, int precision)
{
  type_caster_union d1;
  d1.f = input;
  uint8_t exponent = ((d1.i & 0x7F800000) >> 23) - 127; // mask off the float's sign bit
  int m = 0;
  int sum_m = 0;
  float result = 0;
  int test = (1 << exponent);
  float y = input / test;
  bool max_condition_met = 0;
  uint64_t one = 1;
  uint64_t denom = 0;
  uint64_t prev_denom = 0;
  while((sum_m < precision + 1 && y != 1) || max_condition_met){
    m = 0;
    while((y < 2.f) && (sum_m + m < precision + 1)){
      y *= y;
      m++;
    }

    sum_m += m;
    prev_denom = denom;
    denom = one << sum_m;

    if(sum_m >= precision){ //break when we deliver as much precision as requested
      break;
    }
    if(prev_denom > denom){
      max_condition_met = 1;
      //std::cout << "Warning : unable to provide precision of 2^-" << precision << 
      //             " requested. Providing maximum precision of 2^-64" << std::endl;
      break;
    }

    result += 1.f / (float)denom;
    y /= 2.f;
  }
  return exponent + result;
}

__global__
void compute_log( float* __restrict__ output,
            const float* __restrict__  input,
            int r, int num_inputs, int precision)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_inputs) {
    output[r*num_inputs+i] = binary_log(input[i], precision);
  }
}

void log2_approx (
  std::vector<float> &inputs, 
  std::vector<float> &outputs,
  std::vector<int> &precision,
  const int num_inputs,
  const int precision_count,
  const int repeat)
{
  const int input_size_bytes = num_inputs * sizeof(float);
  const int output_size = num_inputs * precision_count;
  const int output_size_bytes = output_size * sizeof(float);


  float *d_inputs, *d_outputs;
  cudaMalloc((void**)&d_inputs, input_size_bytes);
  cudaMalloc((void**)&d_outputs, output_size_bytes);

  cudaMemcpy(d_inputs, inputs.data(), input_size_bytes, cudaMemcpyHostToDevice); 

  dim3 grid ((num_inputs + 255)/256);
  dim3 block (256);

  for(int i = 0; i < precision_count; ++i) {
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now(); 

    for (int k = 0; k < repeat; ++k) {
      compute_log<<<grid, block>>>(d_outputs, d_inputs, i, num_inputs, precision[i]);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now(); 
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nIterative approximation with " << precision[i] <<" bits of precision\n";
    std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
  }

  cudaMemcpy(outputs.data(), d_outputs, output_size_bytes, cudaMemcpyDeviceToHost); 

  cudaFree(d_inputs); 
  cudaFree(d_outputs);
}
