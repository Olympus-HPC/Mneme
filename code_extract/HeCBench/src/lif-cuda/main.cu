#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

void reference (
    int numNeurons, int neurons_per_item, float dt, 
    float*__restrict__ encode_result,
    float*__restrict__ voltage_array,
    float*__restrict__ reftime_array,
    float tau_rc, float tau_ref,
    float*__restrict__ bias,
    float*__restrict__ gain,
    float*__restrict__ spikes)
{
  for (int i = 0; i < numNeurons; i++)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -expm1f(-dt / tau_rc) * (current - voltage);
    voltage = fmaxf(voltage + dV, 0.f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.f / dt;
    mult += 1.f;

    mult = mult > 1.f ? 1.f : mult;
    mult = mult < 0.f ? 0.f : mult;

    voltage *= mult;

    //printf("%d voltage = %f\n", i, voltage);
    if(voltage > 1.f){
      spike = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage = 0.f;
    }else{
      spike = 0.f;
    }

    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

__global__ void lif (
    int numNeurons, int neurons_per_item, float dt, 
    const float*__restrict__ encode_result,
          float*__restrict__ voltage_array,
          float*__restrict__ reftime_array,
    float tau_rc, float tau_ref,
    const float*__restrict__ bias,
    const float*__restrict__ gain,
          float*__restrict__ spikes)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numNeurons)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = i / neurons_per_item;

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -expm1f(-dt / tau_rc) * (current - voltage);
    voltage = fmaxf(voltage + dV, 0.f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.f / dt;
    mult += 1.f;

    mult = fminf(mult, 1.f);
    mult = fmaxf(mult, 0.f);
    
    voltage *= mult;

    if(voltage > 1.f){
      spike = 1.f / dt;
      ref_time = tau_ref + dt * (1.f - (voltage - 1.f) / dV);
      voltage = 0.f;
    }else{
      spike = 0.f;
    }

    reftime_array[i] = ref_time;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <neurons per item> <num_items> <num_steps>\n", argv[0]);
    return 1;
  }
  const int neurons_per_item = atoi(argv[1]);
  const int num_items = atoi(argv[2]);
  const int num_steps = atoi(argv[3]);

  const int num_neurons = neurons_per_item * num_items;
  const size_t neurons_size = num_neurons * sizeof(float);
  const size_t items_size = num_items * sizeof(float);
  const size_t neurons_per_item_size = neurons_per_item * sizeof(float);

  float dt = 0.1;    // time step
  float tau_rc = 10; // membrane time constant 
  float tau_ref = 2; // refactory time

  float* encode_result = (float*) malloc (items_size);
  float* bias = (float*) malloc (neurons_per_item_size);
  float* gain = (float*) malloc (neurons_per_item_size);

  // test
  float* voltage = (float*) malloc (neurons_size);
  float* reftime = (float*) malloc (neurons_size);
  float* spikes = (float*) malloc (neurons_size);;

  // expected
  float* voltage_gold = (float*) malloc (neurons_size);
  float* reftime_gold = (float*) malloc (neurons_size);
  float* spikes_gold = (float*) malloc (neurons_size);;

  srand(123);
  for (int i = 0; i < num_items; i++) {
    encode_result[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < num_neurons; i++) {
    voltage_gold[i] = voltage[i] = 1.f + rand() / (float)RAND_MAX;
    reftime_gold[i] = reftime[i] = rand() % 5 / 10.f;
  }
  for (int i = 0; i < neurons_per_item; i++) {
    bias[i] = rand() / (float)RAND_MAX;
    gain[i] = rand() / (float)RAND_MAX + 0.5f;
  }

  float* d_encode_result;
  float* d_bias;
  float* d_gain;
  cudaMalloc((void**)&d_encode_result, items_size);
  cudaMalloc((void**)&d_bias, neurons_per_item_size);
  cudaMalloc((void**)&d_gain, neurons_per_item_size);

  // test
  float* d_voltage;
  float* d_reftime;
  float* d_spikes;
  cudaMalloc((void**)&d_voltage, neurons_size);
  cudaMalloc((void**)&d_reftime, neurons_size);
  cudaMalloc((void**)&d_spikes, neurons_size);

  cudaMemcpy(d_encode_result, encode_result, items_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, neurons_per_item_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gain, gain, neurons_per_item_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_voltage, voltage, neurons_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_reftime, reftime, neurons_size, cudaMemcpyHostToDevice);

  dim3 blocks (256);
  dim3 grids ((num_neurons + 255) / 256);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int step = 0; step < num_steps; step++) {
    lif<<<grids, blocks>>>(
        num_neurons, 
        neurons_per_item,
        dt,
        d_encode_result,
        d_voltage,
        d_reftime, 
        tau_rc,
        tau_ref, 
        d_bias,
        d_gain, 
        d_spikes);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (elapsed_time * 1e-3) / num_steps);

  cudaMemcpy(spikes, d_spikes, neurons_size, cudaMemcpyDeviceToHost); 
  cudaMemcpy(voltage, d_voltage, neurons_size, cudaMemcpyDeviceToHost); 
  cudaMemcpy(reftime, d_reftime, neurons_size, cudaMemcpyDeviceToHost); 

  for(int step = 0; step < num_steps; step++) {
    reference(num_neurons, 
        neurons_per_item,
        dt,
        encode_result,
        voltage_gold,
        reftime_gold, 
        tau_rc,
        tau_ref, 
        bias,
        gain, 
        spikes_gold);
  }

  bool ok = true;
  for (int i = 0; i < num_neurons; i++) {
    if (fabsf(spikes[i] - spikes_gold[i]) > 1e-3) {
      printf("@%d: %f %f\n", i, spikes[i], spikes_gold[i]);
      ok = false;
      break;
    }
  }

  free(encode_result);
  free(voltage);
  free(voltage_gold);
  free(reftime);
  free(reftime_gold);
  free(bias);
  free(gain);
  free(spikes);
  free(spikes_gold);

  cudaFree(d_encode_result);
  cudaFree(d_voltage);
  cudaFree(d_reftime);
  cudaFree(d_bias);
  cudaFree(d_gain);
  cudaFree(d_spikes);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
