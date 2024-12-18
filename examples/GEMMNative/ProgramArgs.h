#pragma once
#include <string>

#include "CommandLine/CommandLine.h"

struct Argument
{
	bool isPrintHelp = false;
	int dimMax = 0;
	bool bAlign = true;
};

struct ProgramArgument
{
public:
	ProgramArgument(const std::string& desc);

	void Init();
	void PrintHelp();
	bool Parse(int argc, char* argv[]);

public:
	Argument arg;

private:
	CommandLine cmdLine;
};