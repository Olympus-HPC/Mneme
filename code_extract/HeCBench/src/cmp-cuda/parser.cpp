////////////////////////////////////////////////////////////////////////////////
/**
 * @file parser.inc
 * @date 2015-07-30
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////

#ifndef PARSER_INC
#define PARSER_INC

////////////////////////////////////////////////////////////////////////////////

#include "parser.hpp"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////

std::map<std::string, std::string> parser::_raw_input;

// map <short_form, (help, type)>
std::map<std::string, std::string> parser::_arguments;

std::string parser::_prog_name;

///////////////////////////////////////////////////////////////////////////////

void parser::add_argument(const std::string& short_form,
        const std::string& help)
{
    parser::_arguments.insert(std::pair<std::string, std::string> (short_form, help));
}

///////////////////////////////////////////////////////////////////////////////

const std::string parser::get(const std::string& arg, bool required)
{
    if(required) {
        if(parser::_arguments.count(arg) && parser::_raw_input.count(arg))
            return parser::_raw_input.find(arg)->second;
        else { // if arg required and not found on CLI, print help
            parser::print_help();
            exit(EXIT_FAILURE);
        }
    }
    else {
        if(parser::_arguments.count(arg) && parser::_raw_input.count(arg))
            return parser::_raw_input.find(arg)->second;
    }
    // If argument not found and not required, return default string
    return DEFAULT_STRING;
}

///////////////////////////////////////////////////////////////////////////////

void parser::parse(int argc, const char** argv)
{
    parser::_prog_name = argv[0];

    if(!(argc%2) || argc == 1) { // argc needs to be even and greater than 1
        parser::print_help();
        exit(EXIT_FAILURE);
    }

    for(int i=1; i < argc; i+=2)
    {
        if(!parser::_arguments.count(argv[i]) || !strcmp(argv[i], "-h") || !strcmp(argv[i+1],"-h")) { // if argument not found, print help
            parser::print_help();
            exit(EXIT_FAILURE);
        }
        // Argument successfuly entered
        parser::_raw_input.insert(std::pair<std::string, std::string>(argv[i], argv[i+1]));
    }
}

///////////////////////////////////////////////////////////////////////////////

void parser::print_help()
{
    std::cout << "Usage: " << parser::_prog_name << " [options]" << std::endl;

    std::cout << "Options: " << std::endl;
    for(auto it: parser::_arguments)
    {
        std::cout << "  " << it.first << ": " << it.second << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////

#endif /*! PARSER_INC */

////////////////////////////////////////////////////////////////////////////////
