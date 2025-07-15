#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>
#include "tuningParameters.h"
#include "qtclib.h"
#include "OptionParser.h"
#include "libdata.h"

#include "cudacommon.h"
#define _USE_MATH_DEFINES
#include <float.h>
#include "comm.h"


using namespace std;

#include "kernels_common.h"
#include "kernels_compact_storage.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in megabytes if they are not using a
//   predefined size (i.e. the -s option).
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op){
  op.addOption("PointCount", OPT_INT, "4096", "point count (default: 4096)");
  op.addOption("Threshold", OPT_FLOAT, "1", "cluster diameter threshold (default: 1)");
  op.addOption("SaveOutput", OPT_BOOL, "", "Save output results in files (default: false)");
  op.addOption("Verbose", OPT_BOOL, "", "Print cluster cardinalities (default: false)");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Calls single precision and, if viable, double precision QT-Clustering
//   benchmark.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************
void runTest(const string& name, OptionParser& op);

void RunBenchmark(OptionParser &op){
  runTest("QTC", op);
}

// ****************************************************************************
// Function: calculate_participants
//
// Purpose:
//   This function decides how many GPUs (up to the maximum requested by the user)
//   and threadblocks per GPU will be used. It also returns the total number of
//   thread-blocks across all GPUs and the number of thread-blocks that are in nodes
//   before the current one.
//   In the future, the behavior of this function should be decided based on
//   auto-tuning instead of arbitrary decisions.
//
// Arguments:
//   The number of nodes requested by the user and the four
//   variables that the function computes (passed by reference)
//
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: May 25, 2011
//
// ****************************************************************************
void calculate_participants(int point_count, int node_count, int cwrank,
                            int *thread_block_count,
                            int *total_thread_block_count,
                            int *active_node_count){

  int ac_nd_cnt, thr_blc_cnt, total_thr_blc_cnt;

  ac_nd_cnt = node_count;
  if( point_count <= (node_count-1) * SM_COUNT * GPU_MIN_SATURATION_FACTOR ){
    int K = SM_COUNT * GPU_MIN_SATURATION_FACTOR;
    ac_nd_cnt = (point_count+K-1) / K;
  }

  if( point_count >= ac_nd_cnt * SM_COUNT * OVR_SBSCR_FACTOR ){
    thr_blc_cnt = SM_COUNT * OVR_SBSCR_FACTOR;
    total_thr_blc_cnt = thr_blc_cnt * ac_nd_cnt;
  }else{
    thr_blc_cnt = point_count/ac_nd_cnt;
    if( cwrank < point_count%ac_nd_cnt ){
      thr_blc_cnt++;
    }
    total_thr_blc_cnt = point_count;
  }

  *active_node_count  = ac_nd_cnt;
  *thread_block_count = thr_blc_cnt;
  *total_thread_block_count = total_thr_blc_cnt;

  return;
}

// ****************************************************************************
// Function: runTest
//
// Purpose:
//   This benchmark measures the performance of applying QT-clustering on
//   single precision data.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************

void runTest(const string& name, OptionParser& op)
{
  int matrix_type = 0x0;
  if( 0 == comm_get_rank() ){
    matrix_type |= GLOBAL_MEMORY;

    // find out what type of distance matrix we will be using.
    matrix_type |= COMPACT_STORAGE_MATRIX;
  }
  comm_broadcast ( &matrix_type, 1, COMM_TYPE_INT, 0);

  QTC(name, op, matrix_type);

}

void QTC(const string& name, OptionParser& op, int matrix_type){
  ofstream debug_out, seeds_out;
  void *Ai_mask, *cardnl, *ungrpd_pnts_indr, *clustered_pnts_mask, *result, *dist_to_clust;
  void *indr_mtrx, *degrees;
  int *indr_mtrx_host, *ungrpd_pnts_indr_host, *cardinalities, *output;
  bool save_clusters = false;
  bool be_verbose = false;
  void *distance_matrix_gmem, *distance_matrix;
  float *dist_source, *pnts = NULL;
  float threshold = 1.0f;
  unsigned long int i;
  int max_degree, thread_block_count, total_thread_block_count, active_node_count;
  int cwrank=0, node_count=1, tpb, max_card, iter=0;
  unsigned long int dst_matrix_elems, point_count, max_point_count;

  point_count = op.getOptionInt("PointCount");
  threshold = op.getOptionFloat("Threshold");
  save_clusters = op.getOptionBool("SaveOutput");
  be_verbose = op.getOptionBool("Verbose");


  // TODO - only deal with this size-switch once
  int def_size = op.getOptionInt("size");
  switch( def_size ) {
    case 1:
      // size == 1 should match default values of PointCount,
      // Threshold, TextureMem, and CompactStorage parameters.
      // (i.e., -s 1 is the default)
      point_count    = 4*1024;
      break;
    case 2:
      point_count    = 8*1024;
      break;
    case 3:
      point_count    = 16*1024;
      break;
    case 4:
      point_count    = 16*1024;
      break;
    case 5:
      point_count    = 26*1024;
      break;
    default:
      fprintf( stderr, "unsupported size %d given; terminating\n", def_size );
      return;
  }

  cwrank = comm_get_rank();
  node_count = comm_get_size();

  if( cwrank == 0 ){
    pnts = generate_synthetic_data(&dist_source, &indr_mtrx_host, &max_degree, threshold, point_count, matrix_type);
  }

  comm_broadcast ( &point_count, 1, COMM_TYPE_INT, 0);
  comm_broadcast ( &max_degree, 1, COMM_TYPE_INT, 0);

  dst_matrix_elems = point_count*max_degree;

  if( cwrank != 0 ){ // For all nodes except zero, in a distributed run.
    dist_source = (float*) malloc (sizeof(float)*dst_matrix_elems);
    indr_mtrx_host = (int*) malloc (sizeof(int)*point_count*max_degree);
  }
  // If we need to print the actual clusters later on, we'll need to have all points in all nodes.
  if( save_clusters ){
    if( cwrank != 0 ){
      pnts = (float *)malloc( 2*point_count*sizeof(float) );
    }
    comm_broadcast ( pnts, 2*point_count, COMM_TYPE_FLOAT, 0);
  }

  comm_broadcast ( dist_source, dst_matrix_elems, COMM_TYPE_FLOAT, 0);
  comm_broadcast ( indr_mtrx_host, point_count*max_degree, COMM_TYPE_INT, 0);

  assert( max_degree > 0 );

  calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

  ungrpd_pnts_indr_host = (int*) malloc (sizeof(int)*point_count);
  for(i=0; i<point_count; i++){
    ungrpd_pnts_indr_host[i] = i;
  }

  cardinalities = (int*) malloc (sizeof(int)*2);
  output = (int*) malloc (sizeof(int)*max_degree);

  allocDeviceBuffer(&distance_matrix_gmem, dst_matrix_elems*sizeof(float));
  CHECK_CUDA_ERROR();

  // This is the N*Delta indirection matrix
  allocDeviceBuffer(&indr_mtrx, point_count*max_degree*sizeof(int));

  allocDeviceBuffer(&degrees,             point_count*sizeof(int));
  allocDeviceBuffer(&ungrpd_pnts_indr,    point_count*sizeof(int));
  allocDeviceBuffer(&Ai_mask,             thread_block_count*point_count*sizeof(char));
  allocDeviceBuffer(&dist_to_clust,       thread_block_count*max_degree*sizeof(float));
  allocDeviceBuffer(&clustered_pnts_mask, point_count*sizeof(char));
  allocDeviceBuffer(&cardnl,              thread_block_count*2*sizeof(int));
  allocDeviceBuffer(&result,              point_count*sizeof(int));
#ifdef DEBUG
    int* cardinalities_debug = (int*) malloc (sizeof(int)*thread_block_count*2);
#endif

  // Copy to device, and record transfer time
  copyToDevice(distance_matrix_gmem, dist_source, dst_matrix_elems*sizeof(float));
  copyToDevice(indr_mtrx, indr_mtrx_host, point_count*max_degree*sizeof(int));
  copyToDevice(ungrpd_pnts_indr, ungrpd_pnts_indr_host, point_count*sizeof(int));
  cudaMemset(clustered_pnts_mask, 0, point_count*sizeof(char));
  cudaMemset(dist_to_clust, 0, max_degree*thread_block_count*sizeof(float));

  tpb = ( point_count > THREADSPERBLOCK )? THREADSPERBLOCK : point_count;
  compute_degrees<<<thread_block_count, tpb>>>((int *)indr_mtrx, (int *)degrees, point_count, max_degree);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  // The names of the saved outputs, if enabled, are "p", "p_seeds", and "p."
  if( 0 == cwrank ){
    if( save_clusters ){
      debug_out.open("p");
      for(i=0; i<point_count; i++){
        debug_out << pnts[2*i] << " " << pnts[2*i+1] << endl;
      }
      debug_out.close();
      seeds_out.open("p_seeds");
    }

    cout << "\nInitial ThreadBlockCount: " << thread_block_count;
    cout << " PointCount: " << point_count;
    cout << " Max degree: " << max_degree << "\n" << endl;
    cout.flush();
  }

  max_point_count = point_count;

  tpb = THREADSPERBLOCK;

  distance_matrix = distance_matrix_gmem;

  // Kernel execution
  double qtc_time = 0.0, trim_time = 0.0, update_time = 0.0;

  do{
    stringstream ss;
    int winner_node=-1;
    int winner_index=-1;
    bool this_node_participates = true;

    ++iter;

    calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

    // If there are only a few elements left to cluster, reduce the number of participating nodes (GPUs).
    if( cwrank >= active_node_count ){
      this_node_participates = false;
    }
    comm_update_communicator(cwrank, active_node_count);
    if( !this_node_participates )
      break;
    cwrank = comm_get_rank();

    auto start = std::chrono::steady_clock::now();

    // Main kernel
    QTC_device<<<thread_block_count, tpb>>>((float*)distance_matrix, (char *)Ai_mask, (char *)clustered_pnts_mask,
        (int *)indr_mtrx, (int *)cardnl, (int *)ungrpd_pnts_indr,
        (float *)dist_to_clust, (int *)degrees, point_count, max_point_count,
        max_degree, threshold, cwrank, active_node_count,
        total_thread_block_count);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    qtc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    CHECK_CUDA_ERROR();

#ifdef DEBUG
    printf("cardinalities\n");
    copyFromDevice( cardinalities_debug, cardnl, 2*576*sizeof(int) );
    for (int i = 0; i < 576*2; i++)
     printf("%d %d\n", i, cardinalities_debug[i]);
#endif

    if( thread_block_count > 1 ){
      // We are reducing 128 numbers or less, so one thread should be sufficient.
      reduce_card_device<<<1, 1>>>((int *)cardnl, thread_block_count);
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR();
    }

    copyFromDevice( cardinalities, cardnl, 2*sizeof(int) );
    max_card     = cardinalities[0];
    winner_index = cardinalities[1];

    comm_barrier();

    comm_find_winner(&max_card, &winner_node, &winner_index, cwrank, max_point_count+1);

    if( be_verbose && cwrank == winner_node){ // for non-parallel cases, both "cwrank" and "winner_node" should be zero.
      cout << "[" << cwrank << "] Cluster Cardinality: " << max_card << " (Node: " << cwrank << ", index: " << winner_index << ")" << endl;
    }

    start = std::chrono::steady_clock::now();

    trim_ungrouped_pnts_indr_array<<<1, tpb>>>(winner_index, (int*)ungrpd_pnts_indr, (float*)distance_matrix,
        (int *)result, (char *)Ai_mask, (char *)clustered_pnts_mask,
        (int *)indr_mtrx, (int *)cardnl, (float *)dist_to_clust, (int *)degrees,
        point_count, max_point_count, max_degree, threshold);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();

    end = std::chrono::steady_clock::now();
    trim_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if( cwrank == winner_node){ // for non-parallel cases, these should both be zero.
      if( save_clusters ){
        ss << "p." << iter;
        debug_out.open(ss.str().c_str());
      }

      copyFromDevice(output, (void *)result, max_card*sizeof(int) );

      if( save_clusters ){
        for(int i=0; i<max_card; i++){
          debug_out << pnts[2*output[i]] << " " << pnts[2*output[i]+1] << endl;
        }
        seeds_out << pnts[2*winner_index] << " " << pnts[2*winner_index+1] << endl;
        debug_out.close();
      }
    }

    start = std::chrono::steady_clock::now();

    update_clustered_pnts_mask<<<1, tpb>>>((char *)clustered_pnts_mask, (char *)Ai_mask, max_point_count);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();

    end = std::chrono::steady_clock::now();
    update_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    point_count -= max_card;

  }while( max_card > 1 && point_count );


  if( save_clusters ){
    seeds_out.close();
  }

  if( cwrank == 0){
    cout << "QTC is complete. Clustering iteration count: " << iter << endl;
    cout << "\nKernel execution time\n";
    cout << "qtc: " << qtc_time * 1e-9f << " (s)\n";
    cout << "trim: " << trim_time * 1e-9f << " (s)\n";
    cout << "update: " << update_time * 1e-9f << " (s)\n";
    cout << "total: " << (qtc_time + trim_time + update_time) * 1e-9f << " (s)\n";
    cout.flush();
  }

  free(dist_source);
  free(indr_mtrx_host);
  free(output);
  free(cardinalities);
#ifdef DEBUG
  free(cardinalities_debug);
#endif

  freeDeviceBuffer(distance_matrix_gmem);
  freeDeviceBuffer(indr_mtrx);
  freeDeviceBuffer(degrees);
  freeDeviceBuffer(ungrpd_pnts_indr);
  freeDeviceBuffer(Ai_mask);
  freeDeviceBuffer(dist_to_clust);
  freeDeviceBuffer(clustered_pnts_mask);
  freeDeviceBuffer(cardnl);
  freeDeviceBuffer(result);

  return;
}

////////////////////////////////////////////////////////////////////////////////

void allocDeviceBuffer(void** bufferp, unsigned long bytes)
{
  cudaMalloc(bufferp, bytes);
  CHECK_CUDA_ERROR();
}

void freeDeviceBuffer(void* buffer)
{
  cudaFree(buffer);
}

void copyToDevice(void* to_device, void* from_host, unsigned long bytes)
{
  cudaMemcpy(to_device, from_host, bytes, cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();
}

void copyFromDevice(void* to_host, void* from_device, unsigned long bytes)
{
  cudaMemcpy(to_host, from_device, bytes, cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();
}
