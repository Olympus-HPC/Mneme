#include "dslash.h"

// Global variables
std::vector<int> squaresize(4, LDIM);
size_t sites_on_node = LDIM*LDIM*LDIM*LDIM;
size_t even_sites_on_node = sites_on_node/2;
unsigned int verbose=1;
size_t       warmups=1;

//--------------------------------------------------------------------------------
#include <cassert>
  template<class T>
bool almost_equal(T x, T y, double tol)
{
  return std::abs( x - y ) < tol ;
}

//--------------------------------------------------------------------------------
#include <random>
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<> dis(-1.f, 1.f);

//--------------------------------------------------------------------------------
// initializes su3_matrix
void init_mat(su3_matrix *s) {
  Real r=dis(gen);
  Real i=dis(gen);
  for(int k=0; k<3; ++k){
    for(int l=0; l<3; ++l) {
      s->e[k][l].real=r;
      s->e[k][l].imag=i;
    }
  }
}

//--------------------------------------------------------------------------------
// initializes su3_vector
void init_vec(su3_vector *s) {
  Real r=dis(gen);
  Real i=dis(gen);
  for(int k=0; k<3; ++k) {
    s->c[k].real=r;
    s->c[k].imag=i;
  }
}

//--------------------------------------------------------------------------------
// initialize lattice data
void make_data(su3_vector *src,  su3_matrix *fat, su3_matrix *lng, size_t n) {
  for(size_t i=0;i<n;i++) {
    init_vec(src + i);
    for(int dir=0;dir<4;dir++){
      init_mat(fat + 4*i + dir);
      init_mat(lng + 4*i + dir);
    }
  }
}

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{

  if(argc < 2){
    std::cerr << "Usage <workgroup size>" << std::endl;
    exit(1);
  }

  size_t workgroup_size = atoi(argv[1]);

  size_t iterations = ITERATIONS;
  size_t ldim = LDIM;

  // allocate and initialize the working lattices, matrices, and vectors
  size_t total_sites = sites_on_node;
  std::vector<su3_vector> src(total_sites);
  std::vector<su3_vector> dst(total_sites);
  std::vector<su3_matrix> fat(total_sites*4);
  std::vector<su3_matrix> lng(total_sites*4);
  std::vector<su3_matrix> fatbck(total_sites*4);
  std::vector<su3_matrix> lngbck(total_sites*4);

  size_t *fwd = (size_t *)malloc(sizeof(size_t)*total_sites*4);
  size_t *bck = (size_t *)malloc(sizeof(size_t)*total_sites*4);
  size_t *fwd3 = (size_t *)malloc(sizeof(size_t)*total_sites*4);
  size_t *bck3 = (size_t *)malloc(sizeof(size_t)*total_sites*4);

  // Set up neighbor gather mapping indices
  set_neighbors( fwd, bck, fwd3, bck3 );

  // initialize the data
  make_data(src.data(), fat.data(), lng.data(), total_sites);

  if (verbose > 0) {
    std::cout << "Number of sites = " << ldim << "^4" << std::endl;
    std::cout << "Executing " << iterations << " iterations with " << warmups << " warmups" << std::endl;
    if (workgroup_size != 0)
      std::cout << "Threads per group = " << workgroup_size << std::endl;
    std::cout << std::flush;
  }
  // benchmark call
  const double ttotal = dslash_fn(src, dst, fat, lng, fatbck, lngbck, 
      fwd, bck, fwd3, bck3, 
      iterations, workgroup_size);
  if (verbose > 0)
    std::cout << "Total execution time = " << ttotal << " secs" << std::endl;

  // Validation
  std::vector<su3_vector> chkdst(total_sites);
  if (verbose > 0) {
    std::cout << "Validating the result" << std::endl << std::flush;
  }
  dslash_fn_field(src.data(), chkdst.data(), 1, fat.data(), lng.data(), 
      fatbck.data(), lngbck.data());
  for(size_t i = 0; i < even_sites_on_node; i++){
    for(int k = 0; k < 3; k++){
#ifdef DEBUG
      std::cout << i << " " << k << " " << dst[i].c[k].real << " " << chkdst[i].c[k].real << " " <<
        dst[i].c[k].imag << " " << chkdst[i].c[k].imag << std::endl;
#endif
      assert(almost_equal<Real>(dst[i].c[k].real, chkdst[i].c[k].real, EPISON));
      assert(almost_equal<Real>(dst[i].c[k].imag, chkdst[i].c[k].imag, EPISON));
    }
  }

  // calculate flops/s, etc.
  // each matrix vector multiply is 3*(12 mult + 12 add) = (36 mult + 36 add) = 72 ops
  // sixteen mat vec operations per site 16*72 = 1152
  // plus 15 complex sums = 1152+30 = 1182 ops per site
  const double tflop = (double)iterations * even_sites_on_node * 1182;
  std::cout << "Total GFLOP/s = " << tflop / ttotal / 1.0e9 << std::endl;

  // calculate ideal, per site, data movement for the dslash kernel
  const double memory_usage = (double)even_sites_on_node *
    (sizeof(su3_matrix)*4*4   // 4 su3_matrix reads, per direction
     +sizeof(su3_vector)*16    // 16 su3_vector reads
     +sizeof(size_t)*16        // 16 indirect address reads
     +sizeof(su3_vector));     // 1 su3_vector write
  std::cout << "Total GByte/s (GPU memory) = " << iterations * memory_usage / ttotal / 1.0e9 << std::endl;

  // check memory usage
  if (verbose > 0) {
    const double memory_allocated = (double)total_sites *
      (sizeof(su3_matrix)*4*4   // 4 su3_matrices, each 4x total_sites in size
       +sizeof(su3_vector)*2     // 2 su3_vectors
       +sizeof(size_t)*4*4);     // 4 index arrays with 4 directional dimensions each
    std::cout << "Total allocation for matrices = " << memory_allocated / 1048576.0 << std::endl;
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      std::cout << "Approximate memory usage = " << usage.ru_maxrss/1024.0 << std::endl;
  }

  free(fwd);
  free(bck);
  free(fwd3);
  free(bck3);
  return 0;
}
