#include <iostream>

#include "HelpCuda.h"
#include "ProgramArgs.h"
#include "GEMMBenchmark.h"
#include "KernelHelper.cuh"

ProgramArgument* parseProgramArg(int argc, char* argv[])
{
	std::string desc = "usage: GEMMNative -d <dimMax> -a <align>\
\nGEMMNative -d 100 -a true";

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

	if (pProgArg->arg.dimMax <= 0)
	{
		std::cout << "command arg dim max invalid." << std::endl;
		return nullptr;
	}

	return pProgArg;
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

	if (!Benchmark(pProgArg->arg.dimMax, pProgArg->arg.bAlign))
	{
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}