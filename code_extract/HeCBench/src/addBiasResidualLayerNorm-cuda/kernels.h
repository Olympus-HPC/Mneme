/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

template<typename T>
inline __host__ __device__ float typeToFloat(T a);

template<>
inline __host__ __device__ float typeToFloat(__nv_bfloat16 a) {
  return __bfloat162float(a);
}

template<>
inline __host__ __device__ float typeToFloat(half a) {
  return __half2float(a);
}

template<typename T>
inline __host__ __device__ T floatToType(float a);

template<>
inline __host__ __device__ half floatToType(float a) {
  return __float2half_rn(a);
}

template<>
inline __host__ __device__ __nv_bfloat16 floatToType(float a) {
  return __float2bfloat16(a);
}

template<typename T>
inline __device__ T floatToType2(float a);

// Converts input to half precision in round-to-nearest-even mode
// and populates both halves of half2 with converted value.
template<>
inline __device__ half2 floatToType2(float a) {
  return __float2half2_rn(a);
}

template<>
inline __device__ __nv_bfloat162 floatToType2(float a) {
  return __float2bfloat162_rn(a);
}

template<typename T>
inline __device__ T float2ToType2(float2 a) {
  return a;
}

// Converts both components of float2 to half precision in round-to-nearest mode
// and combines the results into one half2 number.
template<>
inline __device__ half2 float2ToType2(float2 a) {
  return __float22half2_rn(a);
}

template<>
inline __device__ __nv_bfloat162 float2ToType2(float2 a) {
  return __float22bfloat162_rn(a);
}

template<typename T>
inline __device__ float2 type2ToFloat2(T a) {
  return a;
}

template<>
inline __device__ float2 type2ToFloat2(half2 a) {
  return __half22float2(a);
}

template<>
inline __device__ float2 type2ToFloat2(__nv_bfloat162 a) {
  return __bfloat1622float2(a);
}

// Convert between a vector type and a scalar type
template<typename T>
struct TypeConverter {using Type = half2;}; // keep for generality

template<>
struct TypeConverter<__nv_bfloat16> {using Type = __nv_bfloat162;};

// general add
template<typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template<>
inline __device__ half add(half a, half b) {
  return __hadd(a, b);
}

template<>
inline __device__ half2 add(half2 a, half2 b) {
  return __hadd2(a, b);
}

template<>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return __hadd(a, b);
}

template<>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hadd2(a, b);
}

template<typename T>
inline __device__ T add(T a, T b, T c) {
  return a + b + c;
}

template<>
inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return add(add(a, b), c);
}

template<>
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return add(add(a, b), c);
}

// general sub
template<typename T>
inline __device__ T sub(T a, T b) {
  return a - b;
}

template<>
inline __device__ half2 sub(half2 a, half2 b) {
  return __hsub2(a, b);
}

template<>
inline __device__ __nv_bfloat162 sub(__nv_bfloat162 a, __nv_bfloat162 b) {
  return __hsub2(a, b);
}

// general fma
template<typename T>
inline __device__ T fma(T a, T b, T c, T d) {
    return a * b * c + d;
}

template<>
inline __device__ half2 fma(half2 a, half2 b, half2 c, half2 d) {
    return __hadd2(__hmul2(__hmul2(a, b), c), d);
}

template<>
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) {
    return __hadd2(__hmul2(__hmul2(a, b), c), d);
}

// reduction 
template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add(val, __shfl_xor_sync(0xFFFFFFFF, val, mask, 32));
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__global__ void addBiasResidualPostLayerNormV2(
          T* out,
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  using T2             = typename TypeConverter<T>::Type;
  const int        ite = 4;
  const int        tid = threadIdx.x;
  const int        bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;
  T2               local_out_half2[ite];

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  T2 sum = floatToType2<T2>(0.0f);

  // ite = 4 and blockDim.x = n / 8
  // When n = 1024, blockDim.x = 128
  // col_id range: [0-127], [128-255], [256-383], [384-511]
  // block stride = n / 2 = 512
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id         = i * blockDim.x + tid;
    int id             = bid * n / 2 + col_id;
    local_out_half2[i] = add(out_ptr[id], input_ptr[id], bias_ptr[col_id]);
    sum                = add(sum, local_out_half2[i]);
  }

  mean = blockReduceSum<float>(typeToFloat(__hadd(sum.x , sum.y)));
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float var      = 0.0f;
  T2    s_mean_2 = floatToType2<T2>(s_mean);

#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_half2[i] = sub(local_out_half2[i], s_mean_2);
    float v1           = typeToFloat(local_out_half2[i].x);
    float v2           = typeToFloat(local_out_half2[i].y);
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if (tid == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

  T2 s_var_2 = floatToType2<T2>(s_variance);
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id  = i * blockDim.x + tid;
    int id      = bid * n / 2 + col_id;
    out_ptr[id] = fma(local_out_half2[i], s_var_2,
                      gamma_ptr[col_id], beta_ptr[col_id]);
  }
}

template<typename T, int N>
__global__ void addBiasResidualPostLayerNorm(
          T* out, 
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;
  float            local_out_cache[N];

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = typeToFloat(add(out[blockIdx.x * n + idx], input[blockIdx.x * n + idx], bias[idx]));
    mean += local_out;
    // save local_out to local_out_cache to save some recompute
    local_out_cache[i] = local_out;
    idx += blockDim.x;
  }

  mean = blockReduceSum<float>(mean);
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    variance += (local_out - s_mean) * (local_out - s_mean);
    idx += blockDim.x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = variance / n + layernorm_eps;
  }
  __syncthreads();

#pragma unroll N
  for (int idx = threadIdx.x, i = 0; idx < n && i < N; ++i) {
    float local_out = local_out_cache[i];
    out[blockIdx.x * n + idx] =
      floatToType<T>(((local_out - s_mean) * rsqrtf(s_variance)) * typeToFloat(gamma[idx]) + typeToFloat(beta[idx]));
    idx += blockDim.x;
  }
}

template<typename T>
__global__ void generalAddBiasResidualPostLayerNorm(
          T* out, 
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const float layernorm_eps,
    const int n)
{
  using T2 = typename TypeConverter<T>::Type;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float            mean     = 0.0f;
  float            variance = 0.0f;

  T2*       out_ptr   = (T2*)out;
  const T2* input_ptr = (const T2*)input;
  const T2* bias_ptr  = (const T2*)bias;
  const T2* gamma_ptr = (const T2*)gamma;
  const T2* beta_ptr  = (const T2*)beta;

  float local_out = 0.0f;
  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    T2     tmp           = add(add(out_ptr[id], input_ptr[id]), bias_ptr[idx]);
    float2 local_out_fp2 = type2ToFloat2(tmp);
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;
    // save tmp to out_ptr to save some recomputation
    out_ptr[id] = tmp;
  }

  mean = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    variance += (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
    variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
    int    id            = blockIdx.x * n / 2 + idx;
    float2 local_out_fp2 = type2ToFloat2(out_ptr[id]);
    float2 gamma_val     = type2ToFloat2(gamma_ptr[idx]);
    float2 beta_val      = type2ToFloat2(beta_ptr[idx]);
    local_out_fp2.x      = (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y      = (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id]          = float2ToType2<T2>(local_out_fp2);
  }
}
