#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "HelpCuda.h"
#include "Reduction.cuh"

__constant__ int BLOCK_DIM = 128;

__global__ void reduce(float* x, float* y)
{
	// ����ִ�е����۰��Լ����ʱ��ʵ���ϵ��߳�ִ�й��̣�
	// 1.�߳�0-127,offset = N/2��������һ��
	// 1.�߳�0-127,offset = N/4�������ڶ���
	// ...
	// ���˺�����ѭ����ÿһ�ֶ��ᱻ��⡢���䵽�߳̿��ڵ������߳���ִ�У�������һ���߳�����ִ��һ������ѭ��
	const int tid = threadIdx.x;
	float* curX = x + blockIdx.x * blockDim.x;

	// ͨ���߳̿���ͬ�����߳̿�0�еĹ�Լ˳��
	// ��һ�֣�curX[0] += curX[0+64];curX[1] += curX[1+64];...;curX[63] += curX[63+64]
	// �ڶ��֣�curX[0] += curX[0+32];curX[1] += curX[1+32];...;curX[31] += curX[31+32]
	// �����֣�curX[0] += curX[0+16];curX[1] += curX[1+16];...;curX[15] += curX[15+16]
	// �����֣�curX[0] += curX[0+8];curX[1] += curX[1+8];...;curX[7] += curX[7+8]
	// �����֣�curX[0] += curX[0+4];curX[1] += curX[1+4];...;curX[3] += curX[3+4]
	// �����֣�curX[0] += curX[0+2];curX[1] += curX[1+2]
	// �����֣�curX[0] += curX[0+1]
	for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
	{
		// ��������ɸѡ��ʵ�ʵ���ÿ����Ч���߳��������룬��"�߳����ķֻ�"
		// Ҫ�������СΪ�߳̿��С��������
		if (tid < offset)
		{
			// �˺�������"��ָ����߳�"������������ִ��˳�������˳����ܲ�ͬ�������߳�0��1��...127֮��ʵ���ϲ��е�
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