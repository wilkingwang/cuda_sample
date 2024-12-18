#include "ProgramArgs.h"

ProgramArgument::ProgramArgument(const std::string& desc) : cmdLine(desc)
{

}

void ProgramArgument::Init()
{
	cmdLine.addArgument({"-h", "--help"}, &arg.isPrintHelp, "Help Flag");
	cmdLine.addArgument({"-d", "--dimMax"}, &arg.dimMax, "Matrix Max Dim");
	cmdLine.addArgument({"-a", "--align"}, &arg.bAlign, "align");

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