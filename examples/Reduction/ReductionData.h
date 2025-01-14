#pragma once
#include <vector>

class ReductionHostData
{
public:
	ReductionHostData(const int num) : N(num)
	{
	}

	bool Init();

public:
	int N = (int)1e8;
	int blockSize = 0;

	std::vector<float> h_x;
	std::vector<float> h_y;
};

class ReductionGlobalMemSimData
{
public:
	bool Init(const ReductionHostData* hostData);

	bool Run(const ReductionHostData* hostData);

	bool CheckResult(const ReductionHostData* hostData);

	void Destroy();

public:
	float* d_x = nullptr;
	float* d_y = nullptr;
};

class ReductionSharedMemSimData
{
public:
	bool Init(const ReductionHostData* hostData);

	bool Run(const ReductionHostData* hostData);

	bool CheckResult(const ReductionHostData* hostData);

	void Destroy();

public:
	float* d_x = nullptr;
	float* d_y = nullptr;
};