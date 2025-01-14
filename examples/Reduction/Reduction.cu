#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "HelpCuda.h"
#include "Reduction.cuh"

__constant__ int BLOCK_DIM = 128;

__global__ void reduce(float* x, float* y)
{
	// 这里执行迭代折半规约计算时，实际上的线程执行过程：
	// 1.线程0-127,offset = N/2，迭代第一次
	// 1.线程0-127,offset = N/4，迭代第二次
	// ...
	// 即核函数中循环的每一轮都会被拆解、分配到线程块内的所有线程上执行，而不是一个线程连续执行一次完整循环
	const int tid = threadIdx.x;
	float* curX = x + blockIdx.x * blockDim.x;

	// 通过线程块内同步，线程块0中的规约顺序：
	// 第一轮：curX[0] += curX[0+64];curX[1] += curX[1+64];...;curX[63] += curX[63+64]
	// 第二轮：curX[0] += curX[0+32];curX[1] += curX[1+32];...;curX[31] += curX[31+32]
	// 第三轮：curX[0] += curX[0+16];curX[1] += curX[1+16];...;curX[15] += curX[15+16]
	// 第四轮：curX[0] += curX[0+8];curX[1] += curX[1+8];...;curX[7] += curX[7+8]
	// 第五轮：curX[0] += curX[0+4];curX[1] += curX[1+4];...;curX[3] += curX[3+4]
	// 第六轮：curX[0] += curX[0+2];curX[1] += curX[1+2]
	// 第七轮：curX[0] += curX[0+1]
	for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
	{
		// 由于条件筛选，实际导致每轮有效的线程数量减半，即"线程束的分化"
		// 要求数组大小为线程块大小的整数倍
		if (tid < offset)
		{
			// 核函数中是"单指令多线程"，代码真正的执行顺序与出现顺序可能不同，所有线程0、1、...127之间实际上并行的
			curX[tid] += curX[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		y[blockIdx.x] = curX[0];
	}
}

__global__ void reduce_shared(float* d_x, float* d_y, int N)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int n = bid * blockDim.x + tid;

	__shared__ float s_y[128];
	s_y[tid] = (n < N) ? d_x[n] : 0.0f;
	__syncthreads();

	for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			s_y[tid] += s_y[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		d_y[bid] = s_y[0];
	}
}

unsigned int getBlockSize()
{
	int blockSize = 0;

	checkCudaErrors(cudaMemcpyFromSymbol(&blockSize, BLOCK_DIM, sizeof(int)));
	return blockSize;
}

unsigned int blocksForSize(unsigned int n, unsigned int blockSize)
{
	return (n + blockSize - 1) / blockSize;
}

void reduction_global_mem(unsigned int n, unsigned int blockSize, float* d_x, float* d_y)
{
	int gridSize = blocksForSize(n, blockSize);

	reduce << <gridSize, blockSize >> > (d_x, d_y);
}

void reducation_shared_mem(unsigned int n, unsigned int blockSize, float* d_x, float* d_y)
{
	int gridSize = blocksForSize(n, blockSize);

	reduce_shared << <gridSize, blockSize >> > (d_x, d_y, n);
}