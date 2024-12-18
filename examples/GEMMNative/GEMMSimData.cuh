#pragma once
#include <cublas_v2.h>

#include "GEMMHostData.h"

struct GEMMSimData
{
public:
	GEMMSimData(const int dimMax);

	bool Init(const GEMMMatrixData* matrixData);

	bool Run(GEMMMatrixData* matrixData);

	void Destroy();

private:
	bool runSelfHandGEMM(GEMMMatrixData* matrixData);
	bool runBlasGEMM(GEMMMatrixData* matrixData);
	bool checkResult(const GEMMMatrixData* matrixData);

public:
	float alpha = 1.0f;
	float beta = 0.0f;

	int iter = 10;
	double flopsPerMatrixMul = 0;
	double gigaFlops[2] = { 0, 0 };
	double msecPerMatrixMul[2] = { 0, 0 };

	float* d_matrixA = nullptr;
	float* d_matrixB = nullptr;
	float* d_matrixHandOnC = nullptr;
	float* d_matrixBalsC = nullptr;

	cublasHandle_t blasPtr = nullptr;
};