#include<stdio.h>
#include <iostream>
#define EMPTY 0xFFFFFFFF

//TODO: add this in a separate file for definitions
#ifdef DEBUG_GPU
#define DEBUG_PRINT_GPU 1
#endif

#ifdef DEBUG_CPU
#define DEBUG_PRINT_CPU 1
#endif

#define LASSM_MIN_QUAL 10
#define LASSM_MIN_HI_QUAL 20
#define LASSM_MIN_VIABLE_DEPTH 0.2
#define LASSM_MIN_EXPECTED_DEPTH 0.3
#define LASSM_RATING_THRES 0
//#define MAX_WALK_LEN 121+(2*251)
#define LASSM_MIN_KMER_LEN 21
#define LASSM_SHIFT_SIZE 8
#define LASSM_MAX_KMER_LEN 121
#define FULL_MASK 0xffffffff


struct cstr_type{
    char* start_ptr;
    int length;
    __device__ cstr_type(){}
    __device__ cstr_type(char* ptr, int len){
        start_ptr = ptr;
        length = len;
    }

    __device__ bool operator==(const cstr_type& in2){
        bool str_eq = true;
        if(length != EMPTY && in2.length != EMPTY)
            for(int i = 0; i < in2.length; i++){
                if(start_ptr[i] != in2.start_ptr[i]){
                    str_eq = false;
                    break;
                }
            }
        return (str_eq && (length == in2.length));
    }
};

__device__ void cstr_copy(cstr_type& str1, cstr_type& str2);

struct ExtCounts {
  uint32_t count_A;
  uint32_t count_C;
  uint32_t count_G;
  uint32_t count_T;

  __device__ void print(){
    printf("count_A:%d, count_C:%d, count_G:%d, count_T:%d\n", count_A, count_C, count_G, count_T);
  }

  __device__
  void inc(char ext, int count) {
    switch (ext) {
      case 'A':
        atomicAdd(&count_A,count);
        // count_A = (count_A > 65535) ? 65535 : count_A;
        break;
      case 'C':
        atomicAdd(&count_C,count);
        // count_C = (count_C > 65535) ? 65535 : count_C;
        break;
      case 'G':
        atomicAdd(&count_G,count);
        // count_G = (count_G > 65535) ? 65535 : count_G;
        break;
      case 'T':
        atomicAdd(&count_T,count);
        // count_T = (count_T > 65535) ? 65535 : count_T;
        break;
    }
  }
};


  struct MerBase {
    char base;
    uint32_t nvotes_hi_q, nvotes, rating;
    __device__ void print(){
      printf("base:%c, nvotes_hiq_q:%d, nvotes:%d, rating:%d\n", base, nvotes_hi_q, nvotes, rating);
    }

    __device__
    uint16_t get_base_rating(int depth) {
      double min_viable = max(LASSM_MIN_VIABLE_DEPTH * depth, 2.0);
      double min_expected_depth = max(LASSM_MIN_EXPECTED_DEPTH * depth, 2.0);
      if (nvotes == 0) return 0;
      if (nvotes == 1) return 1;
      if (nvotes < min_viable) return 2;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q < min_viable) return 3;
      if (min_expected_depth > nvotes && nvotes >= min_viable && nvotes_hi_q >= min_viable) return 4;
      if (nvotes >= min_expected_depth && nvotes_hi_q < min_viable) return 5;
      if (nvotes >= min_expected_depth && min_viable < nvotes_hi_q && nvotes_hi_q < min_expected_depth) return 6;
      return 7;
    }
  };

struct MerFreqs {

  // how many times this kmer has occurred: don't need to count beyond 65536
  // count of high quality extensions and low quality extensions - structure comes from kmer_dht.hpp
  ExtCounts hi_q_exts, low_q_exts;
  // the final extensions chosen - A,C,G,T, or F,X
  char ext;
  // the count of the final extension
  int count;

    __device__
    bool comp_merbase(MerBase& elem1, MerBase& elem2){
        if(elem1.rating != elem2.rating)
            return elem1.rating > elem2.rating;
        if (elem1.nvotes_hi_q != elem2.nvotes_hi_q)
            return elem1.nvotes_hi_q > elem2.nvotes_hi_q;
        if (elem1.nvotes != elem2.nvotes)
            return elem1.nvotes > elem2.nvotes;

        return true;
    }

    __device__
    void sort_merbase(MerBase (&merbases)[4]){
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                if(comp_merbase(merbases[i], merbases[j])){
                    MerBase temp = merbases[i];
                    merbases[i] = merbases[j];
                    merbases[j] = temp;
                }
            }
        }
    }
  __device__
  void set_ext(int seq_depth) {
    // set extension similarly to how it is done with localassm in mhm
    MerBase mer_bases[4] = {{.base = 'A', .nvotes_hi_q = hi_q_exts.count_A, .nvotes = low_q_exts.count_A},
                            {.base = 'C', .nvotes_hi_q = hi_q_exts.count_C, .nvotes = low_q_exts.count_C},
                            {.base = 'G', .nvotes_hi_q = hi_q_exts.count_G, .nvotes = low_q_exts.count_G},
                            {.base = 'T', .nvotes_hi_q = hi_q_exts.count_T, .nvotes = low_q_exts.count_T}};
    for (int i = 0; i < 4; i++) {
      mer_bases[i].rating = mer_bases[i].get_base_rating(seq_depth);
    }

    // sort bases in descending order of quality
    sort_merbase(mer_bases);

    int top_rating = mer_bases[0].rating;
    int runner_up_rating = mer_bases[1].rating;
    //if (top_rating < runner_up_rating) 
    //DIE("top_rating ", top_rating, " < ", runner_up_rating, "\n");
    //the commented stuff above is handled by the assertion below on GPU
   // assert(top_rating >= runner_up_rating);// for now finding a way around for assertion
    if(top_rating < runner_up_rating)
      printf("******* ASSERTION FAILED IN sort_merbase************");
    int top_rated_base = mer_bases[0].base;
    ext = 'X';
    count = 0;
    // no extension (base = 0) if the runner up is close to the top rating
    // except, if rating is 7 (best quality), then all bases of rating 7 are forks
    if (top_rating > LASSM_RATING_THRES) {         // must have at least minViable bases
      if (top_rating <= 3) {    // must be uncontested
        if (runner_up_rating == 0) ext = top_rated_base;
      } else if (top_rating < 6) {
        if (runner_up_rating < 3) ext = top_rated_base;
      } else if (top_rating == 6) {  // viable and fair hiQ support
        if (runner_up_rating < 4) ext = top_rated_base;
      } else {                     // strongest rating trumps
        if (runner_up_rating < 7) {
          ext = top_rated_base;
        } else {
          if (mer_bases[2].rating == 7 || mer_bases[0].nvotes == mer_bases[1].nvotes) ext = 'F';
          else if (mer_bases[0].nvotes > mer_bases[1].nvotes) ext = mer_bases[0].base;
          else if (mer_bases[1].nvotes > mer_bases[0].nvotes) ext = mer_bases[1].base;
        }
      }
    }
    for (int i = 0; i < 4; i++) {
      if (mer_bases[i].base == ext) {
        count = mer_bases[i].nvotes;
        break;
      }
    }
  }
};


struct loc_ht{
    cstr_type key;
    MerFreqs val;
    __device__ loc_ht(cstr_type in_key, MerFreqs in_val){
      key = in_key;
      val = in_val;
    }
};

struct loc_ht_bool{
    cstr_type key;
    bool val;
    __device__ loc_ht_bool(cstr_type in_key, bool in_val){
      key = in_key;
      val = in_val;
    }
};

__device__ void print_mer(cstr_type& mer);
__global__ void ht_kernel(loc_ht* ht, char* contigs, int* offset_sum, int kmer_size);
__device__ void ht_insert(loc_ht* thread_ht, cstr_type kmer_key, cstr_type ctg_val, uint32_t max_size);
__device__ void ht_delete(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size);
__device__ loc_ht& ht_get(loc_ht* thread_ht, cstr_type kmer_key, uint32_t max_size);
__device__ unsigned hash_func(cstr_type key, uint32_t max_size);
__device__ void count_mers(loc_ht* thrd_loc_ht, char* loc_r_reads, uint32_t max_ht_size, char* loc_r_quals, int32_t* reads_r_offset, int32_t& r_rds_cnt, 
int32_t* rds_count_r_sum, double& loc_ctg_depth, int& mer_len, uint32_t& qual_offset, int64_t& excess_reads, const long int idx);
__device__ char walk_mers(loc_ht* thrd_loc_ht, loc_ht_bool* thrd_ht_bool, uint32_t max_ht_size, int& mer_len, cstr_type& mer_walk_temp, cstr_type& longest_walk, cstr_type& walk, const int idx, int max_walk_len);
__global__ void iterative_walks_kernel(uint32_t* cid, uint32_t* ctg_offsets, char* contigs, char* reads_r, char* quals_r,  uint32_t* reads_r_offset,  uint32_t* rds_count_r_sum, 
double* ctg_depth, loc_ht* global_ht,  uint32_t* prefix_ht, loc_ht_bool* global_ht_bool, int kmer_len, uint32_t max_mer_len_off, uint32_t *term_counts, int64_t num_walks, int64_t max_walk_len, 
int64_t sum_ext, int32_t max_read_size, int32_t max_read_count, uint32_t qual_offset, char* longest_walks, char* mer_walk_temp, uint32_t* final_walk_lens, int tot_ctgs);
