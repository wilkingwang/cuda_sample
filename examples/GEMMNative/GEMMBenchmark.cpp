#include "GEMMBenchmark.h"

GEMMBenchmark::GEMMBenchmark()
{
}

GEMMBenchmark::~GEMMBenchmark()
{
}

bool GEMMBenchmark::Init(const int dimMax, const bool bAlign)
{
	if (!hostData.Init(dimMax, bAlign))
	{
		return false;
	}

	simData = new GEMMSimData(hostData.dimMax);
	if (simData == nullptr)
	{
		return false;
	}

	return true;
}

bool GEMMBenchmark::Run()
{
	for (size_t i = 0; i < hostData.dimSizeArr.size(); i++)
	{
		if (!simData->Init(hostData.matrixArr[i]))
		{
			return false;
		}

		if (!simData->Run(hostData.matrixArr[i]))
		{
			return false;
		}
	}

	return true;
}

bool GEMMBenchmark::Destroy()
{
	simData->Destroy();

	return true;
}

bool Benchmark(const int dimMax, const bool bAlign)
{
	GEMMBenchmark gemmbenchmark;

	if (!gemmbenchmark.Init(dimMax, bAlign))
	{
		return false;
	}

	if (!gemmbenchmark.Run())
	{
		return false;
	}

	return true;
}