#include <cuda_runtime.h>

#include "HelpCuda.h"
#include "GEMMNative.cuh"

const size_t BLOCK_SIZE = 16;
const size_t BLOCK_M = 128; // decide how many thing a thread compute and the amount of shared memory to allocate
const size_t BLOCK_N = 128;
const size_t BLOCK_K = 8;
const size_t BLOCK_M_COMPUTE = BLOCK_M / BLOCK_SIZE;
const size_t BLOCK_N_COMPUTE = BLOCK_N / BLOCK_SIZE;

__global__ void MatMul(const float* A, const float* B, float* C,
	size_t M, size_t N, size_t K, float alpha, float beta)
{
	int tx = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCK_M_COMPUTE;
	int ty = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCK_N_COMPUTE;

	float aa[BLOCK_M_COMPUTE * BLOCK_N_COMPUTE] = {};
	for (size_t i = 0; i < K; i++)
	{
		for (size_t m = 0; m < BLOCK_M_COMPUTE; m++)
		{
			for (size_t n = 0; n < BLOCK_N_COMPUTE; n++)
			{
				aa[m * BLOCK_N_COMPUTE + n] += A[(tx + m) * K + i] * B[i * N + ty + n];
			}
		}
	}

	for (size_t m = 0; m < BLOCK_M_COMPUTE; m++)
	{
		for (size_t n = 0; n < BLOCK_N_COMPUTE; n++)
		{
			C[(tx + m) * N + ty + n] = beta * C[(tx + m) * N + ty + n] + alpha * aa[m * BLOCK_N_COMPUTE + n];
		}
	}
}

void sgemm(size_t M, size_t N, size_t K, float* a, float* b, float* c, float alpha, float beta)
{
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);

	MatMul << <gridDim, blockDim >> > (a, b, c, M, N, K, alpha, beta);
}
