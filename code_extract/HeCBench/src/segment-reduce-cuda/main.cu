#include <chrono>
#include <cstdio>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#define VALUE 1

void segreduce (const size_t num_elements, const int repeat ) {
  printf("num_elements = %zu\n", num_elements);

  int *h_in = new int[num_elements];
  int *h_keys = new int[num_elements];
  for (size_t i = 0; i < num_elements; i++) h_in[i] = VALUE;

  int *d_in, *d_keys;
  cudaMalloc(&d_keys, num_elements * sizeof(int));
  cudaMalloc(&d_in, num_elements * sizeof(int));
  cudaMemcpy(d_in, h_in, num_elements * sizeof(int), cudaMemcpyHostToDevice);

  for (size_t segment_size = 16;
              segment_size <= 16384;
              segment_size = segment_size * 2) {

    // initialize input keys which depend on the segment size
    for (size_t i = 0; i < num_elements; i++) h_keys[i] = i / segment_size;
    cudaMemcpy(d_keys, h_keys, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // allocate output keys and values for each segment size
    const size_t num_segments = num_elements / segment_size;
    int *h_out = new int[num_segments];
    int *d_out, *d_keys_out;
    cudaMalloc(&d_keys_out, num_segments * sizeof(int));
    cudaMalloc(&d_out, num_segments * sizeof(int));

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      thrust::reduce_by_key(thrust::device, d_keys, d_keys + num_elements, d_in,
                            d_keys_out, d_out);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("num_segments = %zu ", num_segments);
    printf("segment_size = %zu ", segment_size);
    printf("Throughput = %f (G/s)\n", 1.f * num_elements * repeat / time);

    cudaMemcpy(h_out, d_out, num_segments * sizeof(int), cudaMemcpyDeviceToHost);

    int correct_segment_sum = 0;
    for (size_t i = 0; i < segment_size; i++) {
      correct_segment_sum += h_in[i];
    }

    int errors = 0;
    for (size_t i = 0; i < num_segments; i++) {
      if (h_out[i] != correct_segment_sum) {
        errors++;
        if (errors < 10) {
          printf("segment %zu has sum %d (expected %d)\n", i,
                 h_out[i], correct_segment_sum);
        }
      }
    }

    if (errors > 0) {
      printf("segmented reduction does not agree with the reference! %d "
             "errors!\n", errors);
    }

    cudaFree(d_out);
    cudaFree(d_keys_out);
    delete[] h_out;
  }

  delete[] h_in;
  delete[] h_keys;
  cudaFree(d_in);
  cudaFree(d_keys);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }
  const int multiplier = atoi(argv[1]);  
  const int repeat = atoi(argv[2]);  

  size_t num_elements = 16384 * size_t(multiplier);
  segreduce(num_elements, repeat);
  return 0;
}
