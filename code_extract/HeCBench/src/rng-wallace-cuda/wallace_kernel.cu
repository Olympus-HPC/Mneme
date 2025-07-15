// ************************************************
// wallace_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Wallace random number generator demonstration code.
//
// Contains code both for simply generating the
// next number, such that can be called by
// the options simulations and also for generatedRandomNumberPoolting
// directly into a memory buffer.
//
// Note that in this code, unlike in the descriptions in the chapter, we have given
// each thread a pair of 4 element transforms to perform, each using
// a slightly different hadamard matrix. The reason behind this is that
// the complexity of computation caused a register shortage when 
// 512 threads were needed (2048 pool/4) which is solved by doubling
// the number of values computed per thread and halving the number
// of threads.
// ************************************************


__device__ void Hadamard4x4a(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = p - t;
	q = q - t;
	r = t - r;
	s = t - s;
}

__device__ void Hadamard4x4b(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = t - p;
	q = t - q;
	r = r - t;
	s = s - t;
}


__global__ void rng_wallace(unsigned m_seed, float *globalPool, float *generatedRandomNumberPool, float *chi2Corrections)
{

  __shared__ float pool[WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE];

	const unsigned lcg_a = 241;
	const unsigned lcg_c = 59;
	const unsigned lcg_m = 256;
	const unsigned mod_mask = lcg_m - 1;

	unsigned offset = __mul24(WALLACE_POOL_SIZE, blockIdx.x);

  #pragma unroll
	for (int i = 0; i < 8; i++)
	  pool[threadIdx.x + WALLACE_NUM_THREADS * i] = globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * i];

	__syncthreads();

	// Loop generating generatedRandomNumberPools repeatedly
	for (int loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
	{

		m_seed = (1664525U * m_seed + 1013904223U) & 0xFFFFFFFF;

		unsigned intermediate_address = __mul24(loop, 8 * WALLACE_TOTAL_NUM_THREADS) + 
			__mul24(8 * WALLACE_NUM_THREADS, blockIdx.x) + threadIdx.x;

		if (threadIdx.x == 0)
			pool[WALLACE_CHI2_OFFSET] = chi2Corrections[__mul24(blockIdx.x, WALLACE_NUM_OUTPUTS_PER_RUN) + loop];
		__syncthreads();
		float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
		for (int i = 0; i < 8; i++)
		{
			generatedRandomNumberPool[intermediate_address + i * WALLACE_NUM_THREADS] = 
				pool[__mul24(i, WALLACE_NUM_THREADS) + threadIdx.x] * chi2CorrAndScale;
		}

		float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
		for (int i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
		{
			unsigned seed = (m_seed + threadIdx.x) & mod_mask;
			__syncthreads();
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin0_0 = pool[((seed << 3))];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin1_0 = pool[((seed << 3) + 1)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin2_0 = pool[((seed << 3) + 2)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin3_0 = pool[((seed << 3) + 3)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin0_1 = pool[((seed << 3) + 4)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin1_1 = pool[((seed << 3) + 5)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin2_1 = pool[((seed << 3) + 6)];
			seed = (__mul24(seed, lcg_a) + lcg_c) & mod_mask;
			rin3_1 = pool[((seed << 3) + 7)];

			__syncthreads();

			Hadamard4x4a(rin0_0, rin1_0, rin2_0, rin3_0);
			pool[0 * WALLACE_NUM_THREADS + threadIdx.x] = rin0_0;
			pool[1 * WALLACE_NUM_THREADS + threadIdx.x] = rin1_0;
			pool[2 * WALLACE_NUM_THREADS + threadIdx.x] = rin2_0;
			pool[3 * WALLACE_NUM_THREADS + threadIdx.x] = rin3_0;

			Hadamard4x4b(rin0_1, rin1_1, rin2_1, rin3_1);
			pool[4 * WALLACE_NUM_THREADS + threadIdx.x] = rin0_1;
			pool[5 * WALLACE_NUM_THREADS + threadIdx.x] = rin1_1;
			pool[6 * WALLACE_NUM_THREADS + threadIdx.x] = rin2_1;
			pool[7 * WALLACE_NUM_THREADS + threadIdx.x] = rin3_1;

			__syncthreads();
		}
	}

	__syncthreads();

  #pragma unroll
	for (int i = 0; i < 8; i++)
	  globalPool[offset + threadIdx.x + WALLACE_NUM_THREADS * i] = pool[threadIdx.x + WALLACE_NUM_THREADS * i];
}

