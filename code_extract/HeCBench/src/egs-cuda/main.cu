/****************************************************************************
 *
 * main.cu, Version 1.0.0 Mon 09 Jan 2012
 *
 * ----------------------------------------------------------------------------
 *
 * CUDA EGS
 * Copyright (C) 2012 CancerCare Manitoba
 *
 * The latest version of CUDA EGS and additional information are available online at 
 * http://www.physics.umanitoba.ca/~elbakri/cuda_egs/ and http://www.lippuner.ca/cuda_egs
 *
 * CUDA EGS is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.                                       
 *                                                                           
 * CUDA EGS is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.                              
 *                                                                           
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * ----------------------------------------------------------------------------
 *
 *   Contact:
 *
 *   Jonas Lippuner
 *   Email: jonas@lippuner.ca 
 *
 ****************************************************************************/

#define CUDA_EGS

#include "EGS.h"
#include "output.c"
#include "media.c"
#include "init.cu"
#include "kernels.cu"


uint read_step_counts(ulong *this_total, ulong *grand_total) {
  clock_t start = clock();
  cudaMemcpy(h_total_step_counts, d_total_step_counts, sizeof(total_step_counts_t), cudaMemcpyDeviceToHost);

  for (uchar i = 0; i < NUM_CAT; i++) {
    this_total[i] = 0;
    for (uchar j = 0; j < SIMULATION_NUM_BLOCKS; j++)
      this_total[i] += (*h_total_step_counts)[j][i];

    grand_total[i] += this_total[i];
  }

  clock_t stop = clock();

  return stop - start;
}

int main(int argc, char **argv) {

  // record start time
  clock_t tic = clock();

  // set RNG seed
  uint seed = 1325631759;

  // check whether GPU device is available
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("CUDA device is not available. Exit.\n");
    return 0;
  }
  
  int GPUId = 0;

  cudaSetDevice(GPUId);

  printf("  Phantom . . . . . . . . . . %s\n", egsphant_file);

  printf("  PEGS4 file  . . . . . . . . %s\n", pegs_file);

  ulong num_histories = 100000000;
  printf("  Histories . . . . . . . . . %zu\n", num_histories);

  printf("  MT parameter file . . . . . %s\n", MT_params_file);

  printf("  Photon xsections  . . . . . %s\n", photon_xsections);

  printf("  Atomic ff file  . . . . . . %s\n", atomic_ff_file);

  printf("  Spectrum file  . . . . . . %s\n", spec_file);

  // write settings
  printf("\nSettings\n");
  printf("  Warps per block . . . . . . %d\n", SIMULATION_WARPS_PER_BLOCK);
  printf("  Blocks per multiprocessor . %d\n", SIMULATION_BLOCKS_PER_MULTIPROC);
  printf("  Iterations outer loop . . . %d\n", SIMULATION_ITERATIONS);
#ifdef USE_ENERGY_SPECTRUM
  printf("  USE_ENERGY_SPECTRUM . . . . enabled\n");
#else
  printf("  USE_ENERGY_SPECTRUM . . . . disabled\n");
#endif
#ifdef DO_LIST_DEPTH_COUNT
  printf("  DO_LIST_DEPTH_COUNT . . . . enabled\n");
#else
  printf("  DO_LIST_DEPTH_COUNT . . . . disabled\n");
#endif

  // perform initialization
  init(seed);

  clock_t tic2 = clock();

  cudaEvent_t start, stop; 
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float elapsed;
  float time_sim = 0.0F;
  float time_sum = 0.0F;
  uint time_copy = 0;

  ulong this_total[NUM_CAT];
  ulong grand_total[NUM_CAT];
  for (uchar i = 0; i < NUM_CAT; i++)
    grand_total[i] = 0;

  bool limit_reached = grand_total[p_new_particle] >= num_histories;
  ulong num_in_progress = 0;
  bool init = true;

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  ulong list_depth = 0;
  ulong num_it = 0;
#endif

  printf("simulation running, wait for ETA...");
  do {
    // do simulation step
    cudaEventRecord(start);
    simulation_step_kernel<<<dim3(SIMULATION_BLOCKS_PER_MULTIPROC, NUM_MULTIPROC), SIMULATION_WARPS_PER_BLOCK * WARP_SIZE>>>(init, limit_reached);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    time_sim += elapsed;
    init = false;

    // sum detector scores
    cudaEventRecord(start);
    sum_detector_scores_kernel<<<SUM_DETECTOR_NUM_BLOCKS, SUM_DETECTOR_WARPS_PER_BLOCK * WARP_SIZE>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    time_sum += elapsed;

    // copy counts from device
    time_copy += read_step_counts(this_total, grand_total);
    ulong num_finished_histories = grand_total[p_new_particle];
    limit_reached = num_finished_histories >= num_histories;

    // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
    cudaMemcpy(h_total_list_depth, d_total_list_depth, sizeof(total_list_depth_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_total_num_inner_iterations, d_total_num_inner_iterations, sizeof(total_num_inner_iterations_t), cudaMemcpyDeviceToHost);
    for (uchar i = 0; i < SIMULATION_NUM_BLOCKS; i++) {
      list_depth += (*h_total_list_depth)[i];
      num_it += (*h_total_num_inner_iterations)[i];
    }
#endif

    // count number of particles in progress
    num_in_progress = 0;
    num_in_progress += this_total[p_cutoff_discard];
    num_in_progress += this_total[p_user_discard];
    num_in_progress += this_total[p_photon_step];
    num_in_progress += this_total[p_rayleigh];
    num_in_progress += this_total[p_compton];
    num_in_progress += this_total[p_photo];
    num_in_progress += this_total[p_pair];
    num_in_progress += this_total[p_new_particle];

    // calculate ETA and display progress
    clock_t tac = clock();
    elapsed = (float)(tac - tic) / (float)CLOCKS_PER_SEC;
    float complete = (float)num_finished_histories / (float)num_histories;
    float eta = elapsed / complete - elapsed;
    if (eta < 0.0F)
      eta = 0.0F;

    printf("\r%zu (%.2f%%) histories started, elapsed time: %.0f, ETA: %.0f   ",
           num_finished_histories, 100.0F * complete, elapsed, eta);

  } while (num_in_progress > 0);
  printf("\r");

  printf("\nSimulation step counts\n");
  printf("  Cutoff discard  . . . . . . %zu\n", grand_total[p_cutoff_discard]);
  printf("  User discard  . . . . . . . %zu\n", grand_total[p_user_discard]);
  printf("  Photon step . . . . . . . . %zu\n", grand_total[p_photon_step]);
  printf("  Rayleigh  . . . . . . . . . %zu\n", grand_total[p_rayleigh]);
  printf("  Compton . . . . . . . . . . %zu\n", grand_total[p_compton]);
  printf("  Photo . . . . . . . . . . . %zu\n", grand_total[p_photo]);
  printf("  Pair  . . . . . . . . . . . %zu\n", grand_total[p_pair]);
  printf("  New particles . . . . . . . %zu\n", grand_total[p_new_particle]);

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  printf("\nDivergence\n");
  printf("  Total different steps . . . %zu\n", list_depth);
  printf("  Total iterations  . . . . . %zu\n", num_it);
  printf("  Average different steps . . %f\n", (double)list_depth / (double)num_it);
  printf("  Average active threads  . . %.2f %% (at least)\n", 100.0 * (double)num_it / (double)list_depth);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // copy results
  clock_t copy_tic = clock();
  uint size_det = h_detector.N.x * h_detector.N.y * sizeof(double);

  double *detector_results_count[NUM_DETECTOR_CAT + 1];
  double *detector_results_energy[NUM_DETECTOR_CAT + 1];
  detector_results_count[NUM_DETECTOR_CAT] = (double*)malloc(size_det);
  detector_results_energy[NUM_DETECTOR_CAT] = (double*)malloc(size_det);

  for (uchar i = 0; i < NUM_DETECTOR_CAT; i++) {
    detector_results_count[i] = (double*)malloc(size_det);
    detector_results_energy[i] = (double*)malloc(size_det);

    // copy detector totals from device
    cudaMemcpy(detector_results_count[i], d_detector_totals_count[i], size_det, cudaMemcpyDeviceToHost);
    cudaMemcpy(detector_results_energy[i], d_detector_totals_energy[i], size_det, cudaMemcpyDeviceToHost);
  }

  total_weights_t h_total_weights;
  cudaMemcpy(h_total_weights, d_total_weights, sizeof(total_weights_t), cudaMemcpyDeviceToHost);

  clock_t copy_tac = clock();
  time_copy += (copy_tac - copy_tic);

  double total_weight = 0.0F;
  for (uchar i = 0; i < SIMULATION_NUM_BLOCKS; i++)
    total_weight += h_total_weights[i];

  double average_weight = total_weight / (double)grand_total[p_new_particle];
  for (uint i = 0; i < h_detector.N.x * h_detector.N.y; i++) {
    double total_count = 0.0F;
    double total_energy = 0.0F;

    for (uchar j = 0; j < NUM_DETECTOR_CAT; j++) {
      detector_results_count[j][i] /= average_weight;
      detector_results_energy[j][i] /= average_weight;

      total_count += detector_results_count[j][i];
      total_energy += detector_results_energy[j][i];
    }

    detector_results_count[NUM_DETECTOR_CAT][i] = total_count;
    detector_results_energy[NUM_DETECTOR_CAT][i] = total_energy;
  }

  write_output("./", "count", detector_results_count);
  write_output("./", "energy", detector_results_energy);

  for (uchar i = 0; i <= NUM_DETECTOR_CAT; i++) {
    free(detector_results_count[i]);
    free(detector_results_energy[i]);
  }

  free_all();

  clock_t tac = clock();

  float time_copy_f = (float)time_copy / (float)CLOCKS_PER_SEC * 1000.0F;
  float total_time = (float)(tac - tic) / (float)CLOCKS_PER_SEC * 1000.0F;
  float init_time = (float)(tic2 - tic) / (float)CLOCKS_PER_SEC * 1000.0F;
  float total_cpu_time = (float)(tac - tic2) / (float)CLOCKS_PER_SEC * 1000.0F;
  float other = total_time - time_copy_f - time_sim - time_sum;

  printf("\nTiming statistics\n");
  printf("  Elapsed time  . . . . . . . %.2f ms (%.2f %%)\n", total_time, 100.0);
  printf("  Total CPU/GPU . . . . . . . %.2f ms (%.2f %%)\n", total_cpu_time, 100.0F * total_cpu_time / total_time);
  printf("  Simulation kernel . . . . . %.2f ms (%.2f %%)\n", time_sim, 100.0F * time_sim / total_time);
  printf("  Summing kernel  . . . . . . %.2f ms (%.2f %%)\n", time_sum, 100.0F * time_sum / total_time);
  printf("  Copying . . . . . . . . . . %.2f ms (%.2f %%)\n", time_copy_f, 100.0F * time_copy_f / total_time);
  printf("  Other . . . . . . . . . . . %.2f ms (%.2f %%)\n", other, 100.0F * other / total_time);
  printf("  Initialization. . . . . . . %.2f ms (%.2f %%)\n", init_time, 100.0F * init_time / total_time);

  printf("\nHistories per ms: %f\n", (float)grand_total[p_new_particle] / total_time);

  return 0;
}

