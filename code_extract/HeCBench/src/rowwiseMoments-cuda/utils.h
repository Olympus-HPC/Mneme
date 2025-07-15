#define WARP_SIZE 32
#define num_threads 256

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE  __host__ __device__
#define DEVICE       __forceinline__ __device__
#else
#define HOST_DEVICE 
#define DEVICE inline
#endif

constexpr int kNumThreads = 256;

template <typename T>
DEVICE T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if defined(__HIPCC__)
  return __shfl_down(value, delta, width);
#else
  return __shfl_down_sync(mask, value, delta, width);
#endif
}

struct Block1D {
  static DEVICE int Tid() { return threadIdx.x; }

  static DEVICE int Warps() {
    return blockDim.x / WARP_SIZE;
  }
};

template <typename T, class ReduceOp>
DEVICE T WarpReduce(T val, const ReduceOp& op) {
  #pragma unroll
  for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp, typename B = Block1D>
DEVICE T
BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % WARP_SIZE;
  const int wid = tid / WARP_SIZE;
  val = WarpReduce(val, op);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = WarpReduce(val, op);
  }
  return val;
}

template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}

  HOST_DEVICE WelfordData(
      scalar_t mean,
      scalar_t m2,
      index_t n,
      scalar_t nf)
    : mean(mean), m2(m2), n(n), nf(nf) {}
};

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOps {
  index_t correction;
  bool take_sqrt;
  public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
        acc.m2 + delta * new_delta,
        new_n,
        new_nf,
    };
  }
  DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
        a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
        // setting acc.n as -1 since acc.n might not be able to represent the count
        // correctly within its range, setting it to -1 to avoid confusion
        -1,
        new_count
    };
  }
  DEVICE res_t project(acc_t acc) const {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? sqrt(var) : var, mean);
    return results;
  }

  DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
          WARP_SHFL_DOWN(acc.mean, offset)
        , WARP_SHFL_DOWN(acc.m2, offset)
        , WARP_SHFL_DOWN(acc.n, offset)
        , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }

  HOST_DEVICE WelfordOps(index_t correction, bool take_sqrt)
    : correction(correction), take_sqrt(take_sqrt) {}
};


