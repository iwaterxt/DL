//io/io.h
// Copyright 2014-11-24   (Author: xutao)
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
# ifndef NNET_NNET_IO_H
# define NNET_NNET_IO_H
# include <iostream>
# include <sstream>
# include <string.h>
# include <stdexcept>
# include <assert.h>
namespace nnet{


//make sure the input string is equal to type
void ExpectToken(std::istream &is, std::string &type );

void ExpectToken(std::istream &is, const char *type );

void Check(const char *token) ;

bool CheckToken(std::istream &is, char type);
	
void ReadToken(std::istream &is, std::string &token);
	
void ReadBasicType(std::istream &is, float* value );

void ReadBasicType(std::istream &is, int* value);

void WriteToken(std::ostream &os, std::string& token); 

void WriteToken(std::ostream &os, const char* token);

void WriteToken(std::ostream &os, const int& value);

void WriteBasicType(std::ostream &os, const float &value);

void WriteBasicType(std::ostream &os, const int &value);


	
}
#endif