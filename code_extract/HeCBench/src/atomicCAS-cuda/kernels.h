// int64 atomic_min
__device__ __forceinline__
long long atomic_min(long long *address, long long val)
{
  long long ret = *address;
  while(val < ret)
  {
    long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
      break;
  }
  return ret;
}

// uint64 atomic_min
__device__ __forceinline__
unsigned long long atomic_min(unsigned long long *address, unsigned long long val)
{
  unsigned long long ret = *address;
  while(val < ret)
  {
    unsigned long long old = ret;
    if((ret = atomicCAS(address, old, val)) == old)
      break;
  }
  return ret;
}

// int64 atomic add
__device__ __forceinline__
long long atomic_add(long long *address, long long val)
{
  long long old, newdbl, ret = *address;
  do {
    old = ret;
    newdbl = old+val;
  } while((ret = (long long)atomicCAS((unsigned long long*)address, (unsigned long long)old, (unsigned long long)newdbl)) != old);
  return ret;
}

// int64 atomic_max
__device__ __forceinline__
long long atomic_max(long long *address, long long val)
{
  long long ret = *address;
  while(val > ret)
  {
    long long old = ret;
    if((ret = (long long)atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
      break;
  }
  return ret;
}

// uint64 atomic_max
__device__ __forceinline__
unsigned long long atomic_max(unsigned long long *address, unsigned long long val)
{
  unsigned long long ret = *address;
  while(val > ret)
  {
    unsigned long long old = ret;
    if((ret = atomicCAS(address, old, val)) == old)
      break;
  }
  return ret;
}

// uint64 atomic add
__device__ __forceinline__
unsigned long long atomic_add(unsigned long long *address, unsigned long long val)
{
  unsigned long long old, newdbl, ret = *address;
  do {
    old = ret;
    newdbl = old+val;
  } while((ret = atomicCAS(address, old, newdbl)) != old);
  return ret;
}

// For all double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN

// double atomic_min
__device__ __forceinline__
double atomic_min(double *address, double val)
{
  unsigned long long ret = __double_as_longlong(*address);
  while(val < __longlong_as_double(ret))
  {
    unsigned long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

// double atomic_max
__device__ __forceinline__
double atomic_max(double *address, double val)
{
  unsigned long long ret = __double_as_longlong(*address);
  while(val > __longlong_as_double(ret))
  {
    unsigned long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

// Double-precision floating point atomic add
__device__ __forceinline__
double atomic_add(double *address, double val)
{
  // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
  unsigned long long *ptr = (unsigned long long *)address;
  unsigned long long old, newdbl, ret = *ptr;
  do {
    old = ret;
    newdbl = __double_as_longlong(__longlong_as_double(old)+val);
  } while((ret = atomicCAS(ptr, old, newdbl)) != old);
  return __longlong_as_double(ret);
}

template <typename T>
__global__ 
void atomicMinDerived (T *res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, (T)i);
}

template <typename T>
__global__ 
void atomicMaxDerived (T *res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_max(res, (T)i);
}

template <typename T>
__global__ 
void atomicAddDerived (T *res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_add(res, (T)i);
}
