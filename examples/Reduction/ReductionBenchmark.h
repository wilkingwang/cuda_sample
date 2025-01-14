#pragma once
#include <vector>

#include "ReductionData.h"

class ReductionBenchmark
{
public:
	virtual bool Init() = 0;

	virtual bool Run(const int iterNum) = 0;

	virtual bool CheckResult() = 0;

	virtual void Destroy() = 0;
};

class ReductionGlobalMemBenchmark : public ReductionBenchmark
{
public:
	ReductionGlobalMemBenchmark(int count);

	~ReductionGlobalMemBenchmark();

	bool Init();

	bool Run(const int iterNum);

	bool CheckResult();

	void Destroy();

private:
	ReductionHostData* hostData = nullptr;
	ReductionGlobalMemSimData simData;
};

class ReductionSharedMemBenchmark : public ReductionBenchmark
{
public:
	ReductionSharedMemBenchmark(int count);

	~ReductionSharedMemBenchmark();

	bool Init();

	bool Run(const int iterNum);

	bool CheckResult();

	void Destroy();

private:
	ReductionHostData* hostData = nullptr;
	ReductionSharedMemSimData simData;
};

class ReductionBenchmarkImp
{
public:
	ReductionBenchmarkImp(const int sceneIdx, const int count);

	~ReductionBenchmarkImp();

	bool Init();

	bool Run(const int iterNum);

	bool CheckResult();

private:
	ReductionBenchmark* reduceBenchmark = nullptr;
};