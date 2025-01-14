#include "ProgramArgs.h"

ProgramArgument::ProgramArgument(const std::string& desc) : cmdLine(desc)
{

}

void ProgramArgument::Init()
{
	cmdLine.addArgument({ "-h", "--help" }, &arg.isPrintHelp, "Help Flag");
	cmdLine.addArgument({ "-c", "--count" }, &arg.count, "Matrix Max Dim");
	cmdLine.addArgument({ "-i", "--iter" }, &arg.count, "Iter Num");
	cmdLine.addArgument({ "-s", "--scene" }, &arg.count, "Scene Index");

	return;
}

void ProgramArgument::PrintHelp()
{
	cmdLine.printHelp();
}

bool ProgramArgument::Parse(int argc, char* argv[])
{
	if (argc < 3)
	{
		arg.isPrintHelp = true;
		return true;
	}

	if (!cmdLine.parse(argc, argv))
	{
		return false;
	}

	return true;
}