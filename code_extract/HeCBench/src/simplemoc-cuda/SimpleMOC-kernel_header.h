#ifndef __SimpleMOC_header
#define __SimpleMOC_header


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<sys/time.h>
#include<stdbool.h>
#include<limits.h>
#include<assert.h>
#include<pthread.h>
#include<unistd.h>

#ifdef OPENMP
#include<omp.h>
#endif


// #define q0[g] simd_vecs[g].s0
// #define q1[g] simd_vecs[g].s1
// #define q2[g]  simd_vecs[g].s2
// #define sigT[g] simd_vecs[g].s3
// #define tau[g] simd_vecs[g].s4
// #define sigT2[g] simd_vecs[g].s5
// #define expVal[g]  simd_vecs[g].s6
// #define reuse[g]   simd_vecs[g].s7
// #define flux_integral[g]  simd_vecs[g].s8
// #define tally[g]  simd_vecs[g].s9
// #define t1[g]  simd_vecs[g].sa
// #define t2[g]  simd_vecs[g].sb
// #define t3[g]  simd_vecs[g].sc
// #define t4[g]  simd_vecs[g].sd

// User inputs
typedef struct{
	int source_2D_regions;
	int source_3D_regions;
	int coarse_axial_intervals;
	int fine_axial_intervals;
	int decomp_assemblies_ax; // Number of subdomains per assembly axially
	long segments;
	int egroups;
	int nthreads;
	int repeat;
	size_t nbytes;
} Input;

// Source Region Structure
typedef struct{
	float * fine_flux;
	float * fine_source;
	float * sigT;
} Source;

// Local SIMD Vector Arrays
typedef struct{
	float * q0;
	float * q1;
	float * q2;
	float * sigT;
	float * tau;
	float * sigT2;
	float * expVal;
	float * reuse;
	float * flux_integral;
	float * tally;
	float * t1;
	float * t2;
	float * t3;
	float * t4;
} SIMD_Vectors;


void attenuate_segment( Input * __restrict I, Source * __restrict S,
		int QSR_id, int FAI_id, float * __restrict state_flux,
		SIMD_Vectors * __restrict simd_vecs);

// init.c
Source * aligned_initialize_sources( Input * I );
Source * initialize_sources( Input * I );
Source* copy_sources( Input * I, Source *S ); 
Input * set_default_input( void );
SIMD_Vectors aligned_allocate_simd_vectors(Input * I);
SIMD_Vectors allocate_simd_vectors(Input * I);
double get_time(void);

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
void print_input_summary(Input * input);
void read_CLI( int argc, char * argv[], Input * input );
void print_CLI_error(void);
void read_input_file( Input * I, char * fname);

// papi.c
void papi_serial_init(void);
void counter_init( int *eventset, int *num_papi_events, Input * I );
void counter_stop( int * eventset, int num_papi_events, Input * I );


#endif
