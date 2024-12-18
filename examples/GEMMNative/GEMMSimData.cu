#include <iomanip>
#include <math.h>
#include <cuda_runtime.h>

#include "HelpCuda.h"
#include "GPUTimer.cuh"
#include "GEMMNative.cuh"
#include "GEMMHostData.h"
#include "GEMMSimData.cuh"

GEMMSimData::GEMMSimData(const int dimMax)
{
	checkCudaErrors(cudaMalloc(&d_matrixA, MAXSIZE(dimMax, float)));
	checkCudaErrors(cudaMalloc(&d_matrixB, MAXSIZE(dimMax, float)));
	checkCudaErrors(cudaMalloc(&d_matrixHandOnC, MAXSIZE(dimMax, float)));
	checkCudaErrors(cudaMalloc(&d_matrixBalsC, MAXSIZE(dimMax, float)));

	checkCudaErrors(cublasCreate(&blasPtr));
}

bool GEMMSimData::Init(const GEMMMatrixData* matrixData)
{
	checkCudaErrors(cudaMemset(d_matrixA, 0, MATRIXSIZE(matrixData->M, matrixData->K, float)));
	checkCudaErrors(cudaMemset(d_matrixB, 0, MATRIXSIZE(matrixData->K, matrixData->N, float)));
	checkCudaErrors(cudaMemset(d_matrixHandOnC, 0, MATRIXSIZE(matrixData->M, matrixData->N, float)));
	checkCudaErrors(cudaMemset(d_matrixBalsC, 0, MATRIXSIZE(matrixData->M, matrixData->N, float)));

	checkCudaErrors(cudaMemcpy(d_matrixA, matrixData->matrixA.data(), MATRIXSIZE(matrixData->M, matrixData->K, float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_matrixB, matrixData->matrixB.data(), MATRIXSIZE(matrixData->K, matrixData->N, float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_matrixHandOnC, matrixData->matrixC.data(), MATRIXSIZE(matrixData->M, matrixData->N, float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_matrixBalsC, matrixData->matrixC.data(), MATRIXSIZE(matrixData->M, matrixData->N, float), cudaMemcpyHostToDevice));

	return true;
}

bool GEMMSimData::Run(GEMMMatrixData* matrixData)
{
	flopsPerMatrixMul = 2.0 * matrixData->M * matrixData->N * matrixData->K;

	if (!runSelfHandGEMM(matrixData))
	{
		return false;
	}

	CCE

	if (!runBlasGEMM(matrixData))
	{
		return false;
	}

	if (!checkResult(matrixData))
	{
		return false;
	}

	return true;
}

void GEMMSimData::Destroy()
{
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixHandOnC);
	cudaFree(d_matrixBalsC);
}

bool GEMMSimData::runSelfHandGEMM(GEMMMatrixData* matrixData)
{
	GPUTimer timer(0);

	// warmup
	sgemm(matrixData->M, matrixData->N, matrixData->K, d_matrixA, d_matrixB, d_matrixHandOnC, alpha, beta);
	checkCudaErrors(cudaMemcpy(matrixData->handOnResult.data(), d_matrixHandOnC, MATRIXSIZE(matrixData->M, matrixData->N, float), cudaMemcpyDeviceToHost));

	//timer.Start();
	//for (size_t i = 0; i < iter; i++)
	//{
	//	sgemm(matrixData->M, matrixData->N, matrixData->K, d_matrixA, d_matrixB, d_matrixHandOnC, alpha, beta);
	//}
	//timer.End();

	msecPerMatrixMul[0] = timer.GetElapse() / iter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	std::cout << "Native GEMM Performance= " << std::fixed << std::setprecision(2) << gigaFlops[0]
		<< " GFlop/s, Time= " << msecPerMatrixMul[0] << " msec, Size= " << flopsPerMatrixMul << " Ops." << std::endl;

	return true;
}

bool GEMMSimData::runBlasGEMM(GEMMMatrixData* matrixData)
{
	GPUTimer timer(0);

	// warmup
	checkCudaErrors(cublasSgemm(blasPtr, CUBLAS_OP_N, CUBLAS_OP_N, (int)matrixData->N, (int)matrixData->M, (int)matrixData->K, &alpha,
		d_matrixA, (int)matrixData->M, d_matrixB, (int)matrixData->K, &beta, d_matrixBalsC, (int)matrixData->N));
	checkCudaErrors(cudaMemcpy(matrixData->blasResult.data(), d_matrixBalsC, MATRIXSIZE(matrixData->M, matrixData->N, float), cudaMemcpyDeviceToHost));

	//timer.Start();
	//for (int i = 0; i < iter; i++)
	//{
	//	checkCudaErrors(cublasSgemm(blasPtr, CUBLAS_OP_N, CUBLAS_OP_N, (int)matrixData->N, (int)matrixData->M, (int)matrixData->K, &alpha,
	//		d_matrixB, (int)matrixData->N, d_matrixA, (int)matrixData->K, &beta, d_matrixBalsC, (int)matrixData->N));
	//}
	//timer.End();

	msecPerMatrixMul[1] = timer.GetElapse() / iter;
	gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
	std::cout << "Blas GEMM Performance= " << std::fixed << std::setprecision(2) << gigaFlops[1]
		<< " GFlop/s, Time= " << msecPerMatrixMul[1] << " msec, Size= " << flopsPerMatrixMul << " Ops." << std::endl;

	return true;
}

bool GEMMSimData::checkResult(const GEMMMatrixData* matrixData)
{
	for (size_t i = 0; i < matrixData->M * matrixData->N; i++)
	{
		if (fabs(matrixData->handOnResult[i] - matrixData->blasResult[i]) > Precision)
		{
			std::cout << "Check gemm result failed." << std::endl;
			return false;
		}
	}

	std::cout << "Check gemm result success." << std::endl;
	return true;
}