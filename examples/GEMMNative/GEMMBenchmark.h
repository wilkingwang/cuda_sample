#pragma once
#include "GEMMHostData.h"
#include "GEMMSimData.cuh"

struct GEMMBenchmark
{
public:
	GEMMBenchmark();

	~GEMMBenchmark();

	bool Init(const int dimMax, const bool bAlign);

	bool Run();

	bool Destroy();

private:
	GEMMHostData hostData;
	GEMMSimData* simData = nullptr;
};

bool Benchmark(const int dimMax, const bool bAlign);