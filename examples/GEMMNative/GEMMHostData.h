#pragma once
#include <vector>

const float Precision = 1e-2f;

#define MATRIXSIZE(M, N, type) (sizeof(type) * M * N)
#define MAXSIZE(dimMax, type) (sizeof(type) * dimMax *dimMax)

struct GEMMMatrixData
{
public:
	GEMMMatrixData(const size_t m, const size_t k, const size_t n);

	void Init();

public:
	size_t M = 0;
	size_t K = 0;
	size_t N = 0;

	std::vector<float> matrixA;
	std::vector<float> matrixB;
	std::vector<float> matrixC;
	std::vector<float> handOnResult;
	std::vector<float> blasResult;
};

struct GEMMHostData
{
public:
	GEMMHostData();

	~GEMMHostData();

	bool Init(const int dimSize, const bool bAlign);

private:
	bool initDimSize(const int dimSize, const bool bAlign);

	bool initHostData();

public:
	int dimMax;
	std::vector<int> dimSizeArr;

	std::vector<GEMMMatrixData*> matrixArr;
};