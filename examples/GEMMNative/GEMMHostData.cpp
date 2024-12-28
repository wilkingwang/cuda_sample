#include "Util.h"
#include "GEMMHostData.h"

GEMMMatrixData::GEMMMatrixData(const size_t m, const size_t k, const size_t n) :
	M(m), K(k), N(n)
{
}

void GEMMMatrixData::Init()
{
	matrixA.resize(M * K);
	matrixB.resize(K * N);
	matrixC.resize(M * N);
	handOnResult.resize(M * N);
	blasResult.resize(M * N);

	GenRandomMatrix(matrixA, M, K);
	GenRandomMatrix(matrixB, K, N);
	GenRandomMatrix(matrixC, M, N);
}

GEMMHostData::GEMMHostData()
{
	dimMax = 0;
}

GEMMHostData::~GEMMHostData()
{

}

bool GEMMHostData::Init(const int dimSize, const bool bAlign)
{
	if (!initDimSize(dimSize, bAlign))
	{
		return false;
	}

	if (!initHostData())
	{
		return false;
	}

	return true;
}

bool GEMMHostData::initDimSize(const int dimSize, const bool bAlign)
{
	// 仅对128大小对其的矩阵，计算结果才和cublas一致
	for (int i = 128; i <= dimSize + 127; i += 128)
	{
		if (!bAlign)
		{
			dimSizeArr.emplace_back(i - 1);
			dimSizeArr.emplace_back(i + 1);
		}

		dimSizeArr.emplace_back(i);
	}

	dimMax = *std::max_element(dimSizeArr.begin(), dimSizeArr.end());
	return true;
}

bool GEMMHostData::initHostData()
{
	matrixArr.resize(dimMax);

	for (size_t i = 0; i < dimSizeArr.size(); i++)
	{
		matrixArr[i] = new GEMMMatrixData(dimSizeArr[i], dimSizeArr[i], dimSizeArr[i]);

		if (matrixArr[i] == nullptr)
		{
			return false;
		}

		matrixArr[i]->Init();
	}

	return true;
}