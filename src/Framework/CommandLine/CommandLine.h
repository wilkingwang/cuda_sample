#pragma once

#include <string>
#include <vector>
#include <variant>
#include <iostream>

class CommandLine
{
public:
	typedef std::variant<int32_t*, uint32_t*, double*, float*, bool*, std::string*> Value;

public:
	explicit CommandLine(const std::string& desc);

	void addArgument(const std::vector<std::string>& flags, const Value& value, const std::string &help);

	void printHelp(std::ostream& os = std::cout) const;

	bool parse(int argc, char *argv[]) const;

private:
	struct Argument
	{
		std::vector<std::string> flags;
		Value					 values;
		std::string				 help;
	};

	std::string desc;
	std::vector<Argument> arguments;
};