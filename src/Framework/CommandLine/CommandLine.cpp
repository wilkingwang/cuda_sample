#include <iomanip>
#include <sstream>
#include <algorithm>

#include "CommandLine.h"

CommandLine::CommandLine(const std::string& desc) : desc(std::move(desc))
{
}

void CommandLine::addArgument(const std::vector<std::string>& flags, const Value& value, const std::string& help)
{
	arguments.emplace_back(Argument{flags, value, help});
}

void CommandLine::printHelp(std::ostream& os) const
{
	os << desc << std::endl;

	uint32_t maxFlagLength = 0;
	for (auto const& argument : arguments)
	{
		uint32_t flagLength = 0;
		for (auto const& flag : argument.flags)
		{
			flagLength += static_cast<uint32_t>(flag.size()) + 2;
		}

		maxFlagLength = std::max(maxFlagLength, flagLength);
	}

	for (auto const& argument : arguments)
	{
		std::string flags;
		for (auto const& flag : argument.flags)
		{
			flags += flag + ", ";
		}

		std::stringstream sstr;
		sstr << std::left << std::setw(maxFlagLength) << flags.substr(0, flags.size() - 2);

		size_t spacePos = 0;
		size_t lineWidth = 0;
		while (spacePos != std::string::npos)
		{
			size_t nextSpacePos = argument.help.find_first_of(' ', spacePos + 1);
			sstr << argument.help.substr(spacePos, nextSpacePos - spacePos);
			lineWidth += nextSpacePos - spacePos;
			spacePos = nextSpacePos;

			if (lineWidth > 60)
			{
				os << "\t" << sstr.str() << std::endl;
				sstr = std::stringstream();
				sstr << std::left << std::setw(maxFlagLength - 1) << " ";
				lineWidth = 0;
			}
		}
	}
}

bool CommandLine::parse(int argc, char* argv[]) const
{
	int i = 1;

	while (i < argc)
	{
		std::string flag(argv[i]);
		std::string value;
		bool valueIsSeparate = false;

		size_t equalPos = flag.find('=');
		if (equalPos != std::string::npos)
		{
			value = flag.substr(equalPos + 1);
			flag = flag.substr(0, equalPos);
		}
		else if (i + 1 < argc)
		{
			value = argv[i + 1];
			valueIsSeparate = true;
		}

		bool foundArgument = false;
		for (auto const& argument : arguments)
		{
			if (std::find(argument.flags.begin(), argument.flags.end(), flag) != std::end(argument.flags))
			{
				foundArgument = true;

				if (std::holds_alternative<bool*>(argument.values))
				{
					if (!value.empty() && value != "true" && value != "false")
					{
						valueIsSeparate = false;
					}

					*std::get<bool*>(argument.values) = (value != "false");
				}
				else if (value.empty())
				{
					std::cout << "Failed to parse command line arguments: "
						"Missing value for argument \"" + flag + "\"!" << std::endl;
					return false;
				}
				else if (std::holds_alternative<std::string*>(argument.values))
				{
					*std::get<std::string*>(argument.values) = value;
				}
				else
				{
					std::visit([&value](auto&& arg) {
						std::stringstream sstr(value);
						sstr >> *arg;
						}, argument.values);
				}

				break;
			}
		}

		if (!foundArgument)
		{
			std::cerr << "Ignoring unknown command line argument \"" << flag << "\"." << std::endl;
		}

		++i;

		if (foundArgument && valueIsSeparate)
		{
			++i;
		}
	}

	return true;
}