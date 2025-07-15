#include <cuda.h>
#include "utils.hpp"
#include "kernel.cu"

void kernel_driver_aa(std::string filename,
    std::vector<std::string> &reads, 
    std::vector<std::string> &contigs,
    short h_scoring_matrix[], 
    short openGap, 
    short extendGap)
{

  unsigned maxContigSize = getMaxLength(contigs);
  unsigned maxReadSize = getMaxLength(reads);
  unsigned totalAlignments = contigs.size(); 

  //std::cout <<"max contig:"<<maxContigSize<<std::endl;
  //std::cout <<"max read:"<<maxReadSize<<std::endl;
  //std::cout <<"total aligns:"<<totalAlignments<<std::endl;

  short* h_ref_begin    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_ref_end      = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_begin  = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_end    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_top_scores   = (short*) malloc (sizeof(short) * totalAlignments);
  unsigned* h_offsetA = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  unsigned* h_offsetB = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  char* h_strA = (char*) malloc (maxContigSize * totalAlignments);
  char* h_strB = (char*) malloc (maxContigSize * totalAlignments);

  short h_encoding_matrix[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    23,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,20,4,3,6,
    13,7,8,9,0,11,10,12,2,0,14,5,
    1,15,16,0,19,17,22,18,21};

  // total number of iterations
  int its = (totalAlignments>20000)?(ceil((float)totalAlignments/20000)):1;
  unsigned NBLOCKS    = totalAlignments;
  unsigned leftOvers    = NBLOCKS % its;
  unsigned stringsPerIt = NBLOCKS / its;
  unsigned maxAlignments = stringsPerIt + leftOvers;

  short* d_ref_start;
  cudaMalloc((void**)&d_ref_start, maxAlignments * sizeof(short));

  short* d_ref_end;
  cudaMalloc((void**)&d_ref_end, maxAlignments * sizeof(short));

  short* d_query_start;
  cudaMalloc((void**)&d_query_start, maxAlignments * sizeof(short));

  short* d_query_end;
  cudaMalloc((void**)&d_query_end, maxAlignments * sizeof(short));

  short* d_scores;
  cudaMalloc((void**)&d_scores, maxAlignments * sizeof(short));

  unsigned* d_offset_ref;
  cudaMalloc((void**)&d_offset_ref, maxAlignments * sizeof(unsigned));

  unsigned* d_offset_query;
  cudaMalloc((void**)&d_offset_query, maxAlignments * sizeof(unsigned));

  char* d_strA;
  cudaMalloc((void**)&d_strA, maxContigSize * maxAlignments * sizeof(char));

  char* d_strB;
  cudaMalloc((void**)&d_strB, maxReadSize * maxAlignments * sizeof(char));

  short* d_encoding_matrix;
  cudaMalloc((void**)&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short));
  cudaMemcpy(d_encoding_matrix, h_encoding_matrix, 
      ENCOD_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice);

  short* d_scoring_matrix;
  cudaMalloc((void**)&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short));
  cudaMemcpy(d_scoring_matrix, h_scoring_matrix, 
      SCORE_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice);

  short* ref_begin    = h_ref_begin;
  short* ref_end      = h_ref_end;  
  short* query_begin  = h_query_begin;
  short* query_end    = h_query_end;
  short* top_scores   = h_top_scores;   

  std::cout<<"Number of loop iterations: " << its << std::endl;

  cudaDeviceSynchronize();
  auto start = NOW;

  for (int perGPUIts = 0; perGPUIts < its; perGPUIts++)
  {
    int  blocksLaunched;
    std::vector<std::string>::const_iterator beginAVec;
    std::vector<std::string>::const_iterator endAVec;
    std::vector<std::string>::const_iterator beginBVec;
    std::vector<std::string>::const_iterator endBVec;

    beginAVec = contigs.begin() + perGPUIts * stringsPerIt;
    beginBVec = reads.begin() + (perGPUIts * stringsPerIt);

    if(perGPUIts == its - 1)
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt + leftOvers;
      endBVec = reads.begin() + ((perGPUIts + 1) * stringsPerIt) + leftOvers;
      blocksLaunched = stringsPerIt + leftOvers;
    }
    else
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt;
      endBVec = reads.begin() +  (perGPUIts + 1) * stringsPerIt;
      blocksLaunched = stringsPerIt;
    }

    std::vector<std::string> sequencesA(beginAVec, endAVec);
    std::vector<std::string> sequencesB(beginBVec, endBVec);
    unsigned running_sum = 0;
    int sequences_per_stream = blocksLaunched;

    for (unsigned int i = 0; i < sequencesA.size(); i++)
    {
      running_sum += sequencesA[i].size();
      h_offsetA[i] = running_sum;//sequencesA[i].size();
      // std::cout << "offsetA_h: " << h_offsetA[i] << std::endl;
    }
    unsigned totalLengthA = h_offsetA[sequencesA.size() - 1];
    // std::cout << "totalLengthA: " << totalLengthA << std::endl;

    running_sum = 0;
    for (unsigned int i = 0; i < sequencesB.size(); i++)
    {
      running_sum += sequencesB[i].size();
      h_offsetB[i] = running_sum; //sequencesB[i].size();
      // std::cout << "offsetB_h: " << h_offsetB[i] << std::endl;
    }
    unsigned totalLengthB = h_offsetB[sequencesB.size() - 1];
    // std::cout << "totalLengthB: " << totalLengthB << std::endl;

    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;

    for (unsigned int i = 0; i < sequencesA.size(); i++)
    {
      char* seqptrA = h_strA + offsetSumA;
      memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
      char* seqptrB = h_strB + offsetSumB;
      memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
      offsetSumA += sequencesA[i].size();
      offsetSumB += sequencesB[i].size();
    }

    cudaMemcpyAsync(d_offset_ref, h_offsetA, sizeof(unsigned) * sequences_per_stream,
        cudaMemcpyHostToDevice, 0);

    cudaMemcpyAsync(d_offset_query, h_offsetB, sizeof(unsigned) * sequences_per_stream,
        cudaMemcpyHostToDevice, 0);

    cudaMemcpyAsync(d_strA, h_strA, sizeof(char) * totalLengthA, 
        cudaMemcpyHostToDevice, 0);

    cudaMemcpyAsync(d_strB, h_strB, sizeof(char) * totalLengthB,
        cudaMemcpyHostToDevice, 0);

    unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
    unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;
    printf("Shared memory bytes: %lu\n", ShmemBytes);
    printf("sequences per stream (CUDA grid size): %d\n", sequences_per_stream);
    printf("minSize (CUDA block size): %d\n", minSize);

    dim3 gws_aa(sequences_per_stream);
    dim3 lws_aa(minSize);

    sequence_aa_kernel<<<gws_aa, lws_aa, ShmemBytes>>> (
        false,
        d_strA,
        d_strB,
        d_offset_ref,
        d_offset_query,
        d_ref_start,
        d_ref_end,
        d_query_start,
        d_query_end,
        d_scores,
        openGap,
        extendGap,
        d_scoring_matrix,
        d_encoding_matrix
        );

    // copyin back end index so that we can find new min
    cudaMemcpyAsync(ref_end, d_ref_end, sizeof(short) * sequences_per_stream,
        cudaMemcpyDeviceToHost, 0);

    cudaMemcpyAsync(query_end, d_query_end, sizeof(short) * sequences_per_stream,
        cudaMemcpyDeviceToHost, 0);

    cudaDeviceSynchronize();

    // find the new largest of smaller lengths
    int newMin = get_new_min_length(ref_end, query_end, blocksLaunched);

    dim3 gws_aa_r(sequences_per_stream);
    dim3 lws_aa_r(newMin);

    sequence_aa_kernel<<<gws_aa_r, lws_aa_r, ShmemBytes>>>(
        true,
        d_strA,
        d_strB,
        d_offset_ref,
        d_offset_query,
        d_ref_start,
        d_ref_end,
        d_query_start,
        d_query_end,
        d_scores,
        openGap,
        extendGap,
        d_scoring_matrix,
        d_encoding_matrix
        );

    cudaMemcpyAsync(ref_begin, d_ref_start, sizeof(short) * sequences_per_stream,
        cudaMemcpyDeviceToHost, 0);

    cudaMemcpyAsync(query_begin, d_query_start, sizeof(short) * sequences_per_stream,
        cudaMemcpyDeviceToHost, 0);

    cudaMemcpyAsync(top_scores, d_scores, sizeof(short) * sequences_per_stream,
        cudaMemcpyDeviceToHost, 0);

    ref_begin += stringsPerIt;
    query_begin += stringsPerIt;
    ref_end += stringsPerIt;
    query_end += stringsPerIt;
    top_scores += stringsPerIt;

  }  // iterations end here

  cudaDeviceSynchronize();
  auto end  = NOW;

  cudaFree(d_ref_start);
  cudaFree(d_ref_end);
  cudaFree(d_query_start);
  cudaFree(d_query_end);
  cudaFree(d_scores);
  cudaFree(d_offset_ref);
  cudaFree(d_offset_query);
  cudaFree(d_strA);
  cudaFree(d_strB);
  cudaFree(d_encoding_matrix);
  cudaFree(d_scoring_matrix);

  std::chrono::duration<double> diff = end - start;
  std::cout << "Total Alignments:" << totalAlignments << "\n" 
            << "Max Reference Size:" << maxContigSize << "\n"
            << "Max Query Size:"<< maxReadSize << "\n" 
            << "Total loop iteration time (seconds):"<< diff.count() << "\n";

  std::ofstream results_file(filename);
  if (results_file.is_open()) {

    for(unsigned int k = 0; k < reads.size(); k++){
      results_file << h_top_scores[k] <<"\t"
                   << h_ref_begin[k] <<"\t"
                   << h_ref_end[k] - 1 <<"\t"
                   << h_query_begin[k] <<"\t"
                   << h_query_end[k] - 1
                   << std::endl;
    }
    results_file.flush();
    results_file.close();
  } else {
    std::cerr << "Error opening the result file "
              << filename << std::endl;
  }

  long long int total_cells = 0;
  for (unsigned int l = 0; l < reads.size(); l++) {
    total_cells += reads.at(l).size()*contigs.at(l).size();
  }

  std::cout << "Total Cells:"<<total_cells<<std::endl;

  free(h_top_scores);
  free(h_ref_begin);
  free(h_ref_end);
  free(h_query_begin);
  free(h_query_end);
  free(h_offsetA);
  free(h_offsetB);
  free(h_strA);
  free(h_strB);
}// end of amino acids kernel
