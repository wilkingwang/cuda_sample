#include<iostream>

#include "ProgramArgs.h"
#include "ReductionBenchmark.h"

ProgramArgument* parseProgramArg(int argc, char* argv[])
{
	std::string desc = "usage: Reducation -c <count>\
\nGEMMNative -c 1000000 -i 100 -s 0";

	ProgramArgument* pProgArg = new ProgramArgument(desc);
	if (pProgArg == nullptr)
	{
		std::cout << "command line construct failed." << std::endl;
		return nullptr;
	}

	pProgArg->Init();
	if (!pProgArg->Parse(argc, argv))
	{
		std::cout << "command line parse failed." << std::endl;
		return nullptr;
	}

	if (pProgArg->arg.isPrintHelp)
	{
		return pProgArg;
	}

	if (pProgArg->arg.count <= 0)
	{
		std::cout << "command arg count invalid." << std::endl;
		return nullptr;
	}

	return pProgArg;
}

bool Benchmark(const int sceneIdx, const int count, const int iterNum)
{
	ReductionBenchmarkImp benchmark(sceneIdx, count);

	if (!benchmark.Init())
	{
		return false;
	}

	if (!benchmark.Run(iterNum))
	{
		return false;
	}

	if (!benchmark.CheckResult())
	{
		return false;
	}

	return true;
}

int main(int argc, char* argv[])
{
	ProgramArgument* pProgArg = parseProgramArg(argc, argv);
	if (pProgArg == nullptr)
	{
		std::cout << "parse program args failed." << std::endl;
		return EXIT_FAILURE;
	}

	if (pProgArg->arg.isPrintHelp)
	{
		pProgArg->PrintHelp();
		return EXIT_FAILURE;
	}

	if (!Benchmark(pProgArg->arg.scene, pProgArg->arg.count, pProgArg->arg.iterNum))
	{
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}