/*
 * CUDA Expectation Maximization with Gaussian Mixture Models
 * Multi-GPU implemenetation using OpenMP
 *
 * Written By: Andrew Pangborn
 * 09/2009
 *
 * Department of Computer Engineering
 * Rochester Institute of Technology
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // for clock(), clock_t, CLOCKS_PER_SEC
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include "gaussian.h"

#include "gaussian_kernel.cu"
#include "cluster.cu"
#include "readData.cu"


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
  int original_num_clusters, desired_num_clusters, ideal_num_clusters;

  // Validate the command-line arguments, parse # of clusters, etc 

  // Don't continue if we had a problem with the program arguments
  if(validateArguments(argc,argv,&original_num_clusters,&desired_num_clusters))
    return 1;

  int num_dimensions;
  int num_events;

  // Read FCS data   
  PRINT("Parsing input file...");
  // This stores the data in a 1-D array with consecutive values being the dimensions from a single event
  // (num_events by num_dimensions matrix)
  float* fcs_data_by_event = readData(argv[2],&num_dimensions,&num_events);   

  if(!fcs_data_by_event) {
    printf("Error parsing input file. This could be due to an empty file ");
    printf("or an inconsistent number of dimensions. Aborting.\n");
    return 1;
  }

  auto start = std::chrono::steady_clock::now();

  clusters_t* clusters = cluster(original_num_clusters, desired_num_clusters, &ideal_num_clusters, 
      num_dimensions, num_events, fcs_data_by_event);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  clusters_t saved_clusters;
  memcpy(&saved_clusters,clusters,sizeof(clusters_t));

  const char* result_suffix = ".results";
  const char* summary_suffix = ".summary";
  int filenamesize1 = strlen(argv[3]) + strlen(result_suffix) + 1;
  int filenamesize2 = strlen(argv[3]) + strlen(summary_suffix) + 1;
  char* result_filename = (char*) malloc(filenamesize1);
  char* summary_filename = (char*) malloc(filenamesize2);
  strcpy(result_filename,argv[3]);
  strcpy(summary_filename,argv[3]);
  strcat(result_filename,result_suffix);
  strcat(summary_filename,summary_suffix);

  PRINT("Summary filename: %s\n",summary_filename);
  PRINT("Results filename: %s\n",result_filename);

  // Open up the output file for cluster summary
  FILE* outf = fopen(summary_filename,"w");
  if(!outf) {
    printf("ERROR: Unable to open file '%s' for writing.\n",argv[3]);
    return -1;
  }

  // Print the clusters with the lowest rissanen score to the console and output file
  for(int c=0; c<ideal_num_clusters; c++) {
    if(ENABLE_PRINT) {
      PRINT("Cluster #%d\n",c);
      printCluster(saved_clusters,c,num_dimensions);
      PRINT("\n\n");
    }

    if(ENABLE_OUTPUT) {
      fprintf(outf,"Cluster #%d\n",c);
      writeCluster(outf,saved_clusters,c,num_dimensions);
      fprintf(outf,"\n\n");
    }
  }
  fclose(outf);

  if(ENABLE_OUTPUT) { 
    // Open another output file for the event level clustering results
    FILE* fresults = fopen(result_filename,"w");

    char header[1000];
    FILE* input_file = fopen(argv[2],"r");
    fgets(header,1000,input_file);
    fclose(input_file);
    fprintf(fresults,"%s",header);

    for(int i=0; i<num_events; i++) {
      for(int d=0; d<num_dimensions-1; d++) {
        fprintf(fresults,"%f,",fcs_data_by_event[i*num_dimensions+d]);
      }
      fprintf(fresults,"%f",fcs_data_by_event[i*num_dimensions+num_dimensions-1]);
      fprintf(fresults,"\t");
      for(int c=0; c<ideal_num_clusters-1; c++) {
        fprintf(fresults,"%f,",saved_clusters.memberships[c*num_events+i]);
      }
      fprintf(fresults,"%f",saved_clusters.memberships[(ideal_num_clusters-1)*num_events+i]);
      fprintf(fresults,"\n");
    }
    fclose(fresults);
  }

  // cleanup host memory
  free(fcs_data_by_event);
  freeCluster(&saved_clusters);

  printf("Execution time of the cluster function %f (s)\n", time * 1e-9f);

  return 0;
}

