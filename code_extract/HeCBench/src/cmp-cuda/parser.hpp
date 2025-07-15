////////////////////////////////////////////////////////////////////////////////
/**
 * @file parser.hpp
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

#ifndef PARSER_HPP
#define PARSER_HPP

////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <cstring>
#include <utility>
#include <map>

///////////////////////////////////////////////////////////////////////////////

/*! \brief Default string to be returned in case of failure */
#define DEFAULT_STRING "0"

///////////////////////////////////////////////////////////////////////////////

/*! \brief Class for parsing the Command Line Interface */
class parser
{
  public:
    /*!
     * \brief Parses the command line interface
     * */
    static void parse(int argc, const char** argv);

    /*!
     * \brief Adds arguments to be parsed.
     *
     * \param short_form short form of the parameter. Ex: "-t", "-a", etc
     * \param help help for the parameter
     * */
    static void add_argument(const std::string& short_form,
        const std::string& help);

    /*!
     * \brief Gets the value of the argument
     *
     * \param arg short form of the parameter. Ex: "-t", "-a", etc
     * \param required true if parameter is required, false otherwise
     * \return string containing the value passed in CLI. If parameter is
     * not required the DEFAULT_STRING will be returned
     * */
    static const std::string get(const std::string& arg, bool required);

  protected:
    /**
     * \brief Prints the help in CLI
     * */
    static void print_help();

  private:
    /*! \brief map< option, value> */
    static std::map<std::string, std::string> _raw_input;

    /*! \brief map <short_form, help> */
    static std::map<std::string, std::string> _arguments;

    /*! \brief argv[0]; */
    static std::string _prog_name;
};

////////////////////////////////////////////////////////////////////////////////

#endif /*! PARSER_HPP */

////////////////////////////////////////////////////////////////////////////////
