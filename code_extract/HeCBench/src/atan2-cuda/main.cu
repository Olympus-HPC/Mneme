/*
 * approximate atan2 evaluations
 *
 * Polynomials were obtained using Sollya scripts (in comments below)
 *
 *
*/

/*
f= atan((1-x)/(1+x))-atan(1);
I=[-1+10^(-4);1.0];
filename="atan.txt";
print("") > filename;
for deg from 3 to 11 do begin
  p = fpminimax(f, deg,[|1,23...|],I, floating, absolute); 
  display=decimal;
  acc=floor(-log2(sup(supnorm(p, f, I, absolute, 2^(-20)))));
  print( "   // degree = ", deg, 
         "  => absolute accuracy is ",  acc, "bits" ) >> filename;
  print("template<> constexpr float approx_atan2f_P<", deg, ">(float x){") >> filename;
  display=hexadecimal;
  print(" return ", horner(p) , ";") >> filename;
  print("}") >> filename;
end;
*/

#include <cstdio>
#include <cmath>
#include <limits>
#include <chrono>
#include <cuda.h>

// float

template <int DEGREE>
__device__ __host__
constexpr float approx_atan2f_P(float x);

// degree =  3   => absolute accuracy is  7 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<3>(float x) {
  return x * (float(-0xf.8eed2p-4) + x * x * float(0x3.1238p-4));
}

// degree =  5   => absolute accuracy is  10 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<5>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ecfc8p-4) + z * (float(0x4.9e79dp-4) + z * float(-0x1.44f924p-4)));
}

// degree =  7   => absolute accuracy is  13 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<7>(float x) {
  auto z = x * x;
  return x * (float(-0xf.fcc7ap-4) + z * (float(0x5.23886p-4) + z * (float(-0x2.571968p-4) + z * float(0x9.fb05p-8))));
}

// degree =  9   => absolute accuracy is  16 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<9>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ff73ep-4) +
              z * (float(0x5.48ee1p-4) +
                   z * (float(-0x2.e1efe8p-4) + z * (float(0x1.5cce54p-4) + z * float(-0x5.56245p-8)))));
}

// degree =  11   => absolute accuracy is  19 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<11>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ffe82p-4) +
              z * (float(0x5.526c8p-4) +
                   z * (float(-0x3.18bea8p-4) +
                        z * (float(0x1.dce3bcp-4) + z * (float(-0xd.7a64ap-8) + z * float(0x3.000eap-8))))));
}

// degree =  13   => absolute accuracy is  21 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<13>(float x) {
  auto z = x * x;
  return x * (float(-0xf.fffbep-4) +
              z * (float(0x5.54adp-4) +
                   z * (float(-0x3.2b4df8p-4) +
                        z * (float(0x2.1df79p-4) +
                             z * (float(-0x1.46081p-4) + z * (float(0x8.99028p-8) + z * float(-0x1.be0bc4p-8)))))));
}

// degree =  15   => absolute accuracy is  24 bits
template <>
__device__ __host__
constexpr float approx_atan2f_P<15>(float x) {
  auto z = x * x;
  return x * (float(-0xf.ffff4p-4) +
              z * (float(0x5.552f9p-4 + z * (float(-0x3.30f728p-4) +
                                             z * (float(0x2.39826p-4) +
                                                  z * (float(-0x1.8a880cp-4) +
                                                       z * (float(0xe.484d6p-8) +
                                                            z * (float(-0x5.93d5p-8) + z * float(0x1.0875dcp-8)))))))));
}

template <int DEGREE>
__device__ __host__
constexpr float unsafe_atan2f_impl(float y, float x) {
  constexpr float pi4f = 3.1415926535897932384626434 / 4;
  constexpr float pi34f = 3.1415926535897932384626434 * 3 / 4;

  auto r = (std::abs(x) - std::abs(y)) / (std::abs(x) + std::abs(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4f : pi34f;
  angle += approx_atan2f_P<DEGREE>(r);

  return ((y < 0)) ? -angle : angle;
}

template <int DEGREE>
__device__ __host__
constexpr float unsafe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>(y, x);
}

template <int DEGREE>
__device__ __host__
constexpr float safe_atan2f(float y, float x) {
  return unsafe_atan2f_impl<DEGREE>(y, ((y == 0.f) & (x == 0.f)) ? 0.2f : x);
}

// integer...
/*
  f= (2^31/pi)*(atan((1-x)/(1+x))-atan(1));
  I=[-1+10^(-4);1.0];
  p = fpminimax(f, [|1,3,5,7,9,11|],[|23...|],I, floating, absolute);
 */

template <int DEGREE>
__device__ __host__
constexpr float approx_atan2i_P(float x);

// degree =  3   => absolute accuracy is  6*10^6
template <>
__device__ __host__
constexpr float approx_atan2i_P<3>(float x) {
  auto z = x * x;
  return x * (-664694912.f + z * 131209024.f);
}

// degree =  5   => absolute accuracy is  4*10^5
template <>
__device__ __host__
constexpr float approx_atan2i_P<5>(float x) {
  auto z = x * x;
  return x * (-680392064.f + z * (197338400.f + z * (-54233256.f)));
}

// degree =  7   => absolute accuracy is  6*10^4
template <>
__device__ __host__
constexpr float approx_atan2i_P<7>(float x) {
  auto z = x * x;
  return x * (-683027840.f + z * (219543904.f + z * (-99981040.f + z * 26649684.f)));
}

// degree =  9   => absolute accuracy is  8000
template <>
__device__ __host__
constexpr float approx_atan2i_P<9>(float x) {
  auto z = x * x;
  return x * (-683473920.f + z * (225785056.f + z * (-123151184.f + z * (58210592.f + z * (-14249276.f)))));
}

// degree =  11   => absolute accuracy is  1000
template <>
__device__ __host__
constexpr float approx_atan2i_P<11>(float x) {
  auto z = x * x;
  return x *
         (-683549696.f + z * (227369312.f + z * (-132297008.f + z * (79584144.f + z * (-35987016.f + z * 8010488.f)))));
}

// degree =  13   => absolute accuracy is  163
template <>
__device__ __host__
constexpr float approx_atan2i_P<13>(float x) {
  auto z = x * x;
  return x * (-683562624.f +
              z * (227746080.f +
                   z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}

template <>
__device__ __host__
constexpr float approx_atan2i_P<15>(float x) {
  auto z = x * x;
  return x * (-683562624.f +
              z * (227746080.f +
                   z * (-135400128.f + z * (90460848.f + z * (-54431464.f + z * (22973256.f + z * (-4657049.f)))))));
}

template <int DEGREE>
__device__ __host__
constexpr int unsafe_atan2i_impl(float y, float x) {
  constexpr long long maxint = (long long)(std::numeric_limits<int>::max()) + 1LL;
  constexpr int pi4 = int(maxint / 4LL);
  constexpr int pi34 = int(3LL * maxint / 4LL);

  auto r = (std::abs(x) - std::abs(y)) / (std::abs(x) + std::abs(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4 : pi34;
  angle += int(approx_atan2i_P<DEGREE>(r));

  return (y < 0) ? -angle : angle;
}

template <int DEGREE>
__device__ __host__
constexpr int unsafe_atan2i(float y, float x) {
  return unsafe_atan2i_impl<DEGREE>(y, x);
}

// short (16bits)
template <int DEGREE>
__device__ __host__
constexpr float approx_atan2s_P(float x);

// degree =  3   => absolute accuracy is  53
template <>
__device__ __host__
constexpr float approx_atan2s_P<3>(float x) {
  auto z = x * x;
  return x * ((-10142.439453125f) + z * 2002.0908203125f);
}

// degree =  5   => absolute accuracy is  7
template <>
__device__ __host__
constexpr float approx_atan2s_P<5>(float x) {
  auto z = x * x;
  return x * ((-10381.9609375f) + z * ((3011.1513671875f) + z * (-827.538330078125f)));
}

// degree =  7   => absolute accuracy is  2
template <>
__device__ __host__
constexpr float approx_atan2s_P<7>(float x) {
  auto z = x * x;
  return x * ((-10422.177734375f) + z * (3349.97412109375f + z * ((-1525.589599609375f) + z * 406.64190673828125f)));
}

// degree =  9   => absolute accuracy is 1
template <>
__device__ __host__
constexpr float approx_atan2s_P<9>(float x) {
  auto z = x * x;
  return x * ((-10428.984375f) + z * (3445.20654296875f + z * ((-1879.137939453125f) +
                                                               z * (888.22314453125f + z * (-217.42669677734375f)))));
}

template <int DEGREE>
__device__ __host__
constexpr short unsafe_atan2s_impl(float y, float x) {
  constexpr int maxshort = (int)(std::numeric_limits<short>::max()) + 1;
  constexpr short pi4 = short(maxshort / 4);
  constexpr short pi34 = short(3 * maxshort / 4);

  auto r = (std::abs(x) - std::abs(y)) / (std::abs(x) + std::abs(y));
  if (x < 0)
    r = -r;

  auto angle = (x >= 0) ? pi4 : pi34;
  angle += short(approx_atan2s_P<DEGREE>(r));

  return (y < 0) ? -angle : angle;
}

template <int DEGREE>
__device__ __host__
constexpr short unsafe_atan2s(float y, float x) {
  return unsafe_atan2s_impl<DEGREE>(y, x);
}

__global__
void compute_f (const int n,
                const float *x,
                const float *y,
                      float *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = safe_atan2f< 3>(vy, vx) +
         safe_atan2f< 5>(vy, vx) +
         safe_atan2f< 7>(vy, vx) +
         safe_atan2f< 9>(vy, vx) +
         safe_atan2f<11>(vy, vx) +
         safe_atan2f<13>(vy, vx) +
         safe_atan2f<15>(vy, vx);
}

__global__
void compute_s (const int n,
                const float *x,
                const float *y,
                      short *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = unsafe_atan2s< 3>(vy, vx) +
         unsafe_atan2s< 5>(vy, vx) +
         unsafe_atan2s< 7>(vy, vx) +
         unsafe_atan2s< 9>(vy, vx);
}

__global__
void compute_i (const int n,
                const float *x,
                const float *y,
                      int *r)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float vy = y[i];
  const float vx = x[i];
  r[i] = unsafe_atan2i< 3>(vy, vx) +
         unsafe_atan2i< 5>(vy, vx) +
         unsafe_atan2i< 7>(vy, vx) +
         unsafe_atan2i< 9>(vy, vx) +
         unsafe_atan2i<11>(vy, vx) +
         unsafe_atan2i<13>(vy, vx) +
         unsafe_atan2i<15>(vy, vx);
}

void reference_f (const int n,
                  const float *x,
                  const float *y,
                        float *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = safe_atan2f< 3>(vy, vx) +
           safe_atan2f< 5>(vy, vx) +
           safe_atan2f< 7>(vy, vx) +
           safe_atan2f< 9>(vy, vx) +
           safe_atan2f<11>(vy, vx) +
           safe_atan2f<13>(vy, vx) +
           safe_atan2f<15>(vy, vx);
  }
}

void reference_s (const int n,
                  const float *x,
                  const float *y,
                        short *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2s< 3>(vy, vx) +
           unsafe_atan2s< 5>(vy, vx) +
           unsafe_atan2s< 7>(vy, vx) +
           unsafe_atan2s< 9>(vy, vx);
  }
}

void reference_i (const int n,
                  const float *x,
                  const float *y,
                        int *r)
{
  for (int i = 0; i < n; i++) {
    const float vy = y[i];
    const float vx = x[i];
    r[i] = unsafe_atan2i< 3>(vy, vx) +
           unsafe_atan2i< 5>(vy, vx) +
           unsafe_atan2i< 7>(vy, vx) +
           unsafe_atan2i< 9>(vy, vx) +
           unsafe_atan2i<11>(vy, vx) +
           unsafe_atan2i<13>(vy, vx) +
           unsafe_atan2i<15>(vy, vx);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of coordinates> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const size_t input_bytes = sizeof(float) * n;
  const size_t output_float_bytes = sizeof(float) * n;
  const size_t output_int_bytes = sizeof(int) * n;
  const size_t output_short_bytes = sizeof(short) * n;

  float *x = (float*) malloc (input_bytes);
  float *y = (float*) malloc (input_bytes);

  float *hf = (float*) malloc (output_float_bytes);
    int *hi = (int*) malloc (output_int_bytes);
  short *hs = (short*) malloc (output_short_bytes);

  // reference
  float *rf = (float*) malloc (output_float_bytes);
    int *ri = (int*) malloc (output_int_bytes);
  short *rs = (short*) malloc (output_short_bytes);

  srand(123);
  for (int i = 0; i < n; i++) {
    x[i] = rand() / (float)RAND_MAX + 1.57f;
    y[i] = rand() / (float)RAND_MAX + 1.57f;
  }
  
  float *dx;
  cudaMalloc((void**)&dx, input_bytes);
  cudaMemcpy(dx, x, input_bytes, cudaMemcpyHostToDevice);

  float *dy;
  cudaMalloc((void**)&dy, input_bytes);
  cudaMemcpy(dy, y, input_bytes, cudaMemcpyHostToDevice);

  float *df;
  cudaMalloc((void**)&df, output_float_bytes);

  int *di;
  cudaMalloc((void**)&di, output_int_bytes);

  short *ds;
  cudaMalloc((void**)&ds, output_short_bytes);

  dim3 grids (n / 256 + 1);
  dim3 blocks (256);

  printf("\n======== output type is f32 ========\n");
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_f <<<grids, blocks>>> (n, dy, dx, df);
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(hf, df, output_float_bytes, cudaMemcpyDeviceToHost);

  reference_f (n, y, x, rf);
  float error = 0;
  for (int i = 0; i < n; i++) {
    if (fabsf(rf[i] - hf[i]) > 1e-3f) {
      error += (ri[i] - hi[i]) * (ri[i] - hi[i]);
    }
  }
  printf("RMSE: %f\n", sqrtf(error / n));

  printf("\n======== output type is i32 ========\n");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_i <<<grids, blocks>>> (n, dy, dx, di);
  cudaDeviceSynchronize();

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(hi, di, output_int_bytes, cudaMemcpyDeviceToHost);

  reference_i (n, y, x, ri);
  error = 0;
  for (int i = 0; i < n; i++) {
    if (abs(ri[i] - hi[i]) > 0) {
      error += (ri[i] - hi[i]) * (ri[i] - hi[i]);
    }
  }
  printf("RMSE: %f\n", sqrtf(error / n));

  printf("\n======== output type is i16 ========\n");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    compute_s <<<grids, blocks>>> (n, dy, dx, ds);
  cudaDeviceSynchronize();

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(hs, ds, output_short_bytes, cudaMemcpyDeviceToHost);

  error = 0;
  reference_s (n, y, x, rs);
  for (int i = 0; i < n; i++) {
    if (abs(rs[i] - hs[i]) > 0) {
      error += (rs[i] - hs[i]) * (rs[i] - hs[i]);
    }
  }
  printf("RMSE: %f\n", sqrtf(error / n));

  cudaFree(df);
  cudaFree(di);
  cudaFree(ds);
  cudaFree(dx);
  cudaFree(dy);
  free(x);
  free(y);
  free(hf);
  free(hi);
  free(hs);
  free(rf);
  free(ri);
  free(rs);
  return 0;
}
