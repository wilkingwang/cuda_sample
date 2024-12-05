#include <cuda_runtime.h>

#include "HelpCuda.h"
#include "KernelHelper.cuh"

void allocateArray(void** devPtr, size_t size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char*)device + offset, host, size, cudaMemcpyHostToDevice));
}

void copyArrayFromDevice(void* host, const void* device, int size)
{
	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}