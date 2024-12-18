#pragma once

#include <chrono>
#include <cuda_runtime.h>

#include "HelpCuda.h"

class GPUTimer
{
public:
	GPUTimer(int devId) : deviceId(devId)
	{
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
	}

	~GPUTimer()
	{
		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
	}

	void Start()
	{
		cudaEventRecord(start);
	}

	void End()
	{
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&elapse, start, stop));
	}

	float GetElapse()
	{
		return elapse;
	}

private:
	float elapse = 0.f;

	int deviceId;
	cudaEvent_t start, stop;
};