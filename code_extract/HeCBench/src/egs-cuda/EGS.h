/****************************************************************************
 *
 * CUDA_EGS.h, Version 1.0.0 Mon 09 Jan 2012
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

#include <stdlib.h>
#include <stdio.h>
#include <cstdarg>
#include <string>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

using namespace std;


/*****************
 * CONFIGURATION *
 *****************/

// If you want to use an energy spectrum for the source, uncomment the following line,
// if you want to use a monoenergetic source, comment the following line.

#define USE_ENERGY_SPECTRUM

// If you want to measure the average number of iterations of the inner loop perfored in one
// iteration of the outer loop, uncomment the following line. This number provides a good
// indication of the average thread idleness, but will probably slighly decrease the overall
// performance of the simulation.

#define DO_LIST_DEPTH_COUNT


// warp size
#define WARP_SIZE 32

// number of multiprocessors
#define NUM_MULTIPROC 80

// The following is the number of warps in each block. This should be large enough to ensure
// a good occupancy, but it is limited by the available registers and shared memory.
#define SIMULATION_WARPS_PER_BLOCK 16

// The following is the number of blocks that are launched for each multiprocessor. There is
// probably no reason for this to be much larger than 1. More blocks require more global memory.
#define SIMULATION_BLOCKS_PER_MULTIPROC 1

// The following is the number of iterations of the outer loop per simulation kernel. A larger
// number will increase the performance of the simulation because fewer kernels launches will
// be necessary, which all have an overhead cost. However, a larger number will also increase 
// the accumulative effect of single precision rounding errors, thus potentially decreasing
// the accuracy of the simulation.
#define SIMULATION_ITERATIONS 32768



/*************
 * CONSTANTS *
 *************/

#define PI                          3.1415926535F
#define ELECTRON_REST_MASS_FLOAT    0.5110034F          // MeV * c^(-2)
#define ELECTRON_REST_MASS_DOUBLE   0.5110034           // MeV * c^(-2)
#define HC_INVERSE                  80.65506856998F     // (hc)^(-1) in (Angstrom * MeV)^(-1)
#define TWICE_HC2                   0.000307444456F     // 2*(hc)^2 in (Angstrom * Mev)^2

// category names
// p primary (never scattered)
// c compton (Compton scattered once)
// r rayleigh (Rayleigh scattered once)
// m multiple (scattered more than once)
// t total (all photons)
const char categories[] = "pcrmt";

// buffer to read or write strings
#define CHARLEN	1024
char charBuffer[CHARLEN];

const char *input_file;
const char *egsphant_file = "./data/EGS_phantom_32.egsphant";
const char *pegs_file = "./data/EGS_phantom.pegs4dat";
const char *MT_params_file = "./data/MTGP_3217_0-8191.bin";
const char *photon_xsections = "./data/si"; 
const char *atomic_ff_file = "./data/pgs4form.dat";
const char *spec_file = "./data/tungsten-80kVp-4mmAl.spectrum";


/**********************************
 * MISCELLANEOUS TYPE DEFINITIONS *
 **********************************/

typedef unsigned char           uchar;

// the different indices of a thread
typedef struct indices {
    uint b;     // index of the block in the grid
    uchar w;    // index of the warp in the block
    uchar t;    // index of the thread in the warp
    uint p;     // index of the particle on the stack
} indices;


/**************************
 * SIMULATION DEFINITIONS *
 **************************/

// the number of blocks used to run the simulation kernel
#define SIMULATION_NUM_BLOCKS (SIMULATION_BLOCKS_PER_MULTIPROC * NUM_MULTIPROC)

// all data of one particle
typedef struct particle_t {
    uchar   status;     // the current (or next) simulation step for this particle
    uchar   reserved;   // currently not used
    char    charge;     // charge of the particle (always 0, since only photons are considered)
    bool    process;    // bool indicating whether the particle needs to perform the current 
                        // simulation step
    float   e;          // energy
    float   wt;         // statistical weight
    uint    region;     // current region
    uint    latch;      // variable for tracking scatter events
    
    // position
    float   x;
    float   y;
    float   z;

    // direction
    float   u;
    float   v;
    float   w;
} particle_t;

// we split up the data for each particle into 16-byte (128-bit) blocks (one uint4) so that we get
// coalesced global memory accesses
typedef struct stack_t {
    
    // 1st block
    uint4   *a;
    /* consists of
    uchar   status;     // 1 byte
    uchar   reserved;   // 1 byte
    char    charge;     // 1 byte
    bool    process;    // 1 byte
    float   e;          // 4 bytes
    float   wt;         // 4 bytes
    uint    region;     // 4 bytes
    */

    // 2nd block
    uint4   *b;
    /* consists of
    uint    latch;      // 4 bytes
    float   x;          // 4 bytes
    float   y;          // 4 bytes
    float   z;          // 4 bytes
    */

    // 3rd block
    uint4   *c;
    /* consists of
    float   u;          // 4 bytes
    float   v;          // 4 bytes
    float   w;          // 4 bytes
    [not used]          // 4 bytes
    */
} stack_t;

stack_t     d_stack;
__constant__ stack_t    stack;

enum particle_status {
    p_cutoff_discard    = 0x00,
    p_user_discard      = 0x01,
    p_photon_step       = 0x02,
    p_rayleigh          = 0x03,
    p_compton           = 0x04,
    p_photo             = 0x05,
    p_pair              = 0x06,
    p_new_particle      = 0x07,
    p_empty             = 0x08 
};

// number of different particle statuses
#define NUM_CAT 9
// number of different detector categories (primary, compton, rayleigh, multiple)
#define NUM_DETECTOR_CAT 4

// number of blocks of the summing kernel
#define SUM_DETECTOR_NUM_BLOCKS (2 * NUM_DETECTOR_CAT)
// warps per block of the summing kernel
#define SUM_DETECTOR_WARPS_PER_BLOCK 32


// list depth counter
#ifdef DO_LIST_DEPTH_COUNT
__shared__ uint list_depth_shared[SIMULATION_WARPS_PER_BLOCK];
__shared__ uint num_inner_iterations_shared[SIMULATION_WARPS_PER_BLOCK];
typedef ulong total_list_depth_t[SIMULATION_NUM_BLOCKS];
typedef ulong total_num_inner_iterations_t[SIMULATION_NUM_BLOCKS];
total_list_depth_t *d_total_list_depth, *h_total_list_depth;
total_num_inner_iterations_t *d_total_num_inner_iterations, *h_total_num_inner_iterations;
__constant__ total_list_depth_t *total_list_depth;
__constant__ total_num_inner_iterations_t *total_num_inner_iterations;
#endif

__shared__ uint step_counters_shared[SIMULATION_WARPS_PER_BLOCK][NUM_CAT];
__shared__ double combined_weight_list_shared[SIMULATION_WARPS_PER_BLOCK];
__shared__ float weight_list_shared[SIMULATION_WARPS_PER_BLOCK][WARP_SIZE];

typedef float *detector_scores_t[SIMULATION_NUM_BLOCKS][NUM_DETECTOR_CAT];
typedef double total_weights_t[SIMULATION_NUM_BLOCKS];
typedef ulong total_step_counts_t[SIMULATION_NUM_BLOCKS][NUM_CAT];

detector_scores_t d_detector_scores_count, d_detector_scores_energy;
double *d_detector_totals_count[NUM_DETECTOR_CAT], *d_detector_totals_energy[NUM_DETECTOR_CAT];
total_weights_t *d_total_weights;
total_step_counts_t *d_total_step_counts, *h_total_step_counts;

__constant__ detector_scores_t detector_scores_count, detector_scores_energy;
__constant__ double *detector_totals_count[NUM_DETECTOR_CAT], *detector_totals_energy[NUM_DETECTOR_CAT];
__constant__ total_weights_t *total_weights;
__constant__ total_step_counts_t *total_step_counts;


/********************************
 * MERSENNE TWISTER DEFINITIONS *
 ********************************/

#define MT_EXP 3217
// number of elements in the status array
#define MT_N (MT_EXP / 32 + 1)
// next larger multiple of WARP_SIZE
#define MT_NUM_STATUS (((MT_N - 1) / WARP_SIZE + 1) * WARP_SIZE)
// number of random numbers that each thread can use until a status update is necessary
#define MT_NUM_PER_THREAD (MT_N / WARP_SIZE)
#define MT_TABLE_SIZE 16

typedef struct MT_input_param {
    uint    mexp;
    uint    bit_size;
    uint    id;
    uint    M;              // also called pos in the MTGP code
    uint    sh1;
    uint    sh2;
    uint    tbl[4];
    uint    tmp_tbl[4];
    uint    mask;
} MT_input_param;

typedef struct MT_param {
    uint    M;
    uint    sh1;
    uint    sh2;
    uint    mask;
} MT_param;

typedef struct MT_tables_t {
    uint    recursion[MT_TABLE_SIZE];
    uint    tempering[MT_TABLE_SIZE];
} MT_tables_t;

__shared__  MT_param    MT_params_shared[SIMULATION_WARPS_PER_BLOCK];
__shared__  uint        MT_statuses_shared[SIMULATION_WARPS_PER_BLOCK][MT_N];
__shared__  MT_tables_t MT_tables_shared[SIMULATION_WARPS_PER_BLOCK];
__shared__  uchar       rand_idx_shared[SIMULATION_WARPS_PER_BLOCK];
__shared__  float       random_array_shared[SIMULATION_WARPS_PER_BLOCK][WARP_SIZE * MT_NUM_PER_THREAD];

MT_param    *h_MT_params, *d_MT_params;
uint		*h_MT_statuses, *d_MT_statuses;
MT_tables_t *d_MT_tables;

__constant__ uint *MT_statuses;
__constant__ MT_param *MT_params;
__constant__ MT_tables_t *MT_tables;



/************************
 * GEOMETRY DEFINITIONS *
 ************************/

// source
typedef struct source_t {
#ifdef USE_ENERGY_SPECTRUM
    uint    n;
    float   *xi, *wi;
    int     *bin;
#else
	float   energy;
#endif
    float3  source_point;
    float   rectangle_z;
    float2  rectangle_min;
    float2  rectangle_max;
    float2  rectangle_size;
    float   rectangle_area;
} source_t; 

typedef struct detector_t {
    float3  center;
    float2  d;
    uint2   N;
} detector_t;

typedef struct phantom_t {
    uint3   N;
    float   *x_bounds;
    float   *y_bounds;
    float   *z_bounds;
} phantom_t;

detector_t  h_detector;
source_t    h_source;
phantom_t   h_phantom;

__constant__    detector_t  detector;
__constant__    source_t    source;
__constant__    phantom_t   phantom;



/*******************************
 * SIMULATION DATA DEFINITIONS *
 *******************************/

#define BOUND_COMPTON_MASK 0x000EU
#define VACUUM 0xFFFFU
#define VACUUM_STEP 1E8F
#define EPSGMFP 1E-5F
#define SMALL_POLAR_ANGLE_THRESHOLD 1E-20F

enum region_flags {
    f_rayleigh              = 0x0001U,                  // 0000 0000 0000 0001
    f_bound_compton         = 0x0002U,                  // 0000 0000 0000 0010
    f_bound_compton_2       = 0x0006U,                  // 0000 0000 0000 0110
    f_bound_compton_3       = 0x000AU,                  // 0000 0000 0000 1010
    f_bound_compton_4       = 0x000EU,                  // 0000 0000 0000 1110
    f_atomic_relaxation     = 0x0010U,                  // 0000 0000 0001 0000
    f_photo_electron_angular_distribution = 0x0020U,    // 0000 0000 0010 0000
    f_range_rejection       = 0x0040U                   // 0000 0000 0100 0000
};

typedef struct __align__(16) region_data_t {
    ushort  med;
    ushort  flags;
    float   rhof;
    float   pcut;
    float   ecut;
} region_data_t;

region_data_t *d_region_data;
__constant__ region_data_t *region_data;



/**************************
 * MEDIA DATA DEFINITIONS *
 **************************/

const char *data_dir;


