template<typename divisor_type>
__global__ void throughput_test(
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    int dummy,
    int * buf)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int x1 = x / d1;
  int x2 = x / d2;
  int x3 = x / d3;
  int aggregate = x1 + x2 + x3;  
  if (aggregate & dummy == 1) buf[0] = aggregate;
}

template<typename divisor_type>
__global__ void latency_test(
    divisor_type d1,
    divisor_type d2,
    divisor_type d3,
    divisor_type d4,
    divisor_type d5,
    divisor_type d6,
    divisor_type d7,
    divisor_type d8,
    divisor_type d9,
    divisor_type d10,
    int dummy,
    int * buf)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  x /= d1;
  x /= d2;
  x /= d3;
  x /= d4;
  x /= d5;
  x /= d6;
  x /= d7;
  x /= d8;
  x /= d9;
  x /= d10;
  if (x & dummy == 1) buf[0] = x;
}

__global__ void check(int_fastdiv divisor, int * results)
{
  int divident = blockIdx.x * blockDim.x + threadIdx.x;

  int quotient = divident / (int)divisor;
  int fast_quotient = divident / divisor;

  if (quotient != fast_quotient)
  {
    int error_id = atomicAdd(&results[0], 1);
    if (error_id == 0)
    {
      results[1] = divident;
      results[2] = quotient;
      results[3] = fast_quotient;
    }
  }

  divident = -divident;
  quotient = divident / (int)divisor;
  fast_quotient = divident / divisor;

  if (quotient != fast_quotient)
  {
    int error_id = atomicAdd(&results[0], 1);
    if (error_id == 0)
    {
      results[1] = divident;
      results[2] = quotient;
      results[3] = fast_quotient;
    }
  }
}
