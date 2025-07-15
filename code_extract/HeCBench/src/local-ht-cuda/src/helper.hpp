
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#define CUDA_CHECK(ans)                                                                  \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

struct timer{
double total_time = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> time_begin;
std::chrono::time_point<std::chrono::high_resolution_clock> time_end;
std::chrono::duration<double> diff;

void timer_start(){
    time_begin = std::chrono::high_resolution_clock::now();
}

void timer_end(){
    time_end = std::chrono::high_resolution_clock::now();
}

double get_total_time(){
    diff = time_end - time_begin;
    return diff.count();
}


};

struct ReadSeq {
  std::string read_id;
  std::string seq;
  std::string quals;
};

struct CtgWithReads {
  int32_t cid;
  std::string seq;
  double depth;
  int max_reads;
  std::vector<ReadSeq> reads_left;
  std::vector<ReadSeq> reads_right;
};




std::vector<std::string> read_fasta(std::string in_file, int &largest);
void read_locassm_data(std::vector<CtgWithReads> *data_in, std::string fname, 
uint32_t& max_ctg_size, uint32_t& total_r_reads, uint32_t& total_l_reads, uint32_t& max_read_size, uint32_t& max_r_count, uint32_t& max_l_count);
//templated functions needs to be in the same translation unit
template<typename T>
void print_log(T _log){
    std::cout<<_log<<std::endl;
}

template<typename T>
void print_vals(T val){
    print_log(val);
}

template<typename T, typename... Types>
void print_vals(T val, Types... val_){
    if(sizeof...(val_) == 0){
        print_vals(val);
    }else{
        print_vals(val);
        print_vals(val_...);
        }
}

void print_loc_data(std::vector<CtgWithReads> *data_in);

inline void revcomp(char* str, char* str_rc, int size) {
  int size_rc = 0;
  for (int i = size - 1; i >= 0; i--) {
    switch (str[i]) {
      case 'A': str_rc[size_rc]= 'T'; break;
      case 'C': str_rc[size_rc]= 'G'; break;
      case 'G': str_rc[size_rc]= 'C'; break;
      case 'T': str_rc[size_rc]= 'A'; break;
      case 'N': str_rc[size_rc]= 'N'; break;
      case 'U': case 'R': case 'Y': case 'K': case 'M': case 'S': case 'W': case 'B': case 'D': case 'H': case 'V':
        str_rc[size_rc]= 'N';
        break;
      default:
        print_vals("Illegal char", str[i], "in revcomp of ");
    }
    size_rc++;
  }
}

inline std::string revcomp(std::string instr) {
  std::string str_rc;
  for (int i = instr.size() - 1; i >= 0; i--) {
    switch (instr[i]) {
      case 'A': str_rc += 'T'; break;
      case 'C': str_rc += 'G'; break;
      case 'G': str_rc += 'C'; break;
      case 'T': str_rc += 'A'; break;
      case 'N': str_rc += 'N'; break;
      case 'U': case 'R': case 'Y': case 'K': case 'M': case 'S': case 'W': case 'B': case 'D': case 'H': case 'V':
        str_rc += 'N';
        break;
      default:
        print_vals("Illegal char", instr[i], "in revcomp of ");
    }
  }
  return str_rc;
}
