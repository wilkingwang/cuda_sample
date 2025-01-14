#include <cuda_runtime.h>

#include "HelpCuda.h"
#include "Reduction.cuh"
#include "ReductionData.h"

bool ReductionHostData::Init()
{
	blockSize = getBlockSize();

	h_x.resize(N);
	h_y.resize(blocksForSize(N, blockSize));

	for (size_t i = 0; i < N; i++)
	{
		h_x[i] = 1.23f;
	}

	return true;
}

bool ReductionGlobalMemSimData::Init(const ReductionHostData* hostData)
{
	checkCudaErrors(cudaMalloc(&d_x, sizeof(float) * hostData->N));
	checkCudaErrors(cudaMalloc(&d_y, sizeof(float) * hostData->h_y.size()));

	checkCudaErrors(cudaMemcpy(d_x, hostData->h_x.data(), sizeof(float) * hostData->N, cudaMemcpyDefault));

	return true;
}

bool ReductionGlobalMemSimData::Run(const ReductionHostData* hostData)
{
	reduction_global_mem(hostData->N, hostData->blockSize, d_x, d_y);

	return true;
}

bool ReductionGlobalMemSimData::CheckResult(const ReductionHostData* hostData)
{
	std::vector<float> h_y;

	h_y.resize(hostData->h_y.size());

	CCE
	checkCudaErrors(cudaMemcpy(h_y.data(), d_y, sizeof(float) * h_y.size(), cudaMemcpyDeviceToHost));
	return true;
}

void ReductionGlobalMemSimData::Destroy()
{
	cudaFree(d_x);
	cudaFree(d_y);
}

bool ReductionSharedMemSimData::Init(const ReductionHostData* hostData)
{
	checkCudaErrors(cudaMalloc(&d_x, sizeof(float) * hostData->N));
	checkCudaErrors(cudaMalloc(&d_y, sizeof(float) * hostData->h_y.size()));

	checkCudaErrors(cudaMemcpy(d_x, hostData->h_x.data(), sizeof(float) * hostData->N, cudaMemcpyDefault));

	return true;
}

bool ReductionSharedMemSimData::Run(const ReductionHostData* hostData)
{
	reducation_shared_mem(hostData->N, hostData->blockSize, d_x, d_y);

	return true;
}

bool ReductionSharedMemSimData::CheckResult(const ReductionHostData* hostData)
{
	std::vector<float> h_y;

	h_y.resize(hostData->h_y.size());

	CCE
		checkCudaErrors(cudaMemcpy(h_y.data(), d_y, sizeof(float) * h_y.size(), cudaMemcpyDeviceToHost));
	return true;
}

void ReductionSharedMemSimData::Destroy()
{
	cudaFree(d_x);
	cudaFree(d_y);
}