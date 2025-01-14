#include "HelpCuda.h"
#include "GPUTimer.cuh"

#include "ReductionBenchmark.h"

ReductionGlobalMemBenchmark::ReductionGlobalMemBenchmark(int count)
{
	hostData = new ReductionHostData(count);
}

ReductionGlobalMemBenchmark::~ReductionGlobalMemBenchmark()
{
}

bool ReductionGlobalMemBenchmark::Init()
{
	if (!hostData->Init())
	{
		return false;
	}

	if (!simData.Init(hostData))
	{
		return false;
	}

	return true;
}

bool ReductionGlobalMemBenchmark::Run(const int iterNum)
{
	GPUTimer timer(0);

	timer.Start();
	for (int i = 0; i < iterNum; i++)
	{
		if (!simData.Run(hostData))
		{
			return false;
		}
	}
	timer.End();

	std::cout << "Reduction global mem time cost: " << timer.GetElapse() / iterNum << "us." << std::endl;
	return true;
}

bool ReductionGlobalMemBenchmark::CheckResult()
{
	if (!simData.CheckResult(hostData))
	{
		return false;
	}

	return true;
}

void ReductionGlobalMemBenchmark::Destroy()
{
	delete hostData;
	simData.Destroy();
}

ReductionSharedMemBenchmark::ReductionSharedMemBenchmark(int count)
{
	hostData = new ReductionHostData(count);
}

ReductionSharedMemBenchmark::~ReductionSharedMemBenchmark()
{

}

bool ReductionSharedMemBenchmark::Init()
{
	if (!hostData->Init())
	{
		return false;
	}

	if (!simData.Init(hostData))
	{
		return false;
	}

	return true;
}

bool ReductionSharedMemBenchmark::Run(const int iterNum)
{
	GPUTimer timer(0);

	timer.Start();
	for (int i = 0; i < iterNum; i++)
	{
		if (!simData.Run(hostData))
		{
			return false;
		}
	}
	timer.End();

	std::cout << "Reduction shared mem time cost: " << timer.GetElapse() / iterNum << "us." << std::endl;
	return true;
}

bool ReductionSharedMemBenchmark::CheckResult()
{
	if (!simData.CheckResult(hostData))
	{
		return false;
	}

	return true;
}

void ReductionSharedMemBenchmark::Destroy()
{
	delete hostData;
	simData.Destroy();
}

ReductionBenchmarkImp::ReductionBenchmarkImp(const int sceneIdx, const int count)
{
	if (sceneIdx == 0)
	{
		reduceBenchmark = new ReductionGlobalMemBenchmark(count);
	}

	if (sceneIdx == 1)
	{
		reduceBenchmark = new ReductionSharedMemBenchmark(count);
	}
}

ReductionBenchmarkImp::~ReductionBenchmarkImp()
{
	reduceBenchmark->Destroy();
}

bool ReductionBenchmarkImp::Init()
{
	if (!reduceBenchmark->Init())
	{
		return false;
	}

	return true;
}

bool ReductionBenchmarkImp::Run(const int iterNum)
{
	if (!reduceBenchmark->Run(iterNum))
	{
		return false;
	}

	return true;
}

bool ReductionBenchmarkImp::CheckResult()
{
	if (!reduceBenchmark->CheckResult())
	{
		return false;
	}

	return true;
}