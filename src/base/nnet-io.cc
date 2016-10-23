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
#include "nnet-io.h"

namespace nnet{


//make sure the input string is equal to type
void ExpectToken(std::istream &is, std::string &type ){
	std::string s;
	is >> s ;

	assert(s.size()== type.size());
	for(unsigned int i = 0; i < s.size(); i++)
		assert(s[i]==type[i]);
}

void ExpectToken(std::istream &is, const char *type ){
	std::string s;
	is >> s ;
	assert(s.size()== strlen(type));
	for(unsigned int i = 0; i < s.size(); i++)
		assert(s[i]==type[i]);
}

void Check(const char *token) {
  	assert(*token != '\0');  // check it's nonempty.
  	while (*token != '\0') {
    		assert(!::isspace(*token));
    		token++;
  	}
}

bool CheckToken(std::istream &is, char type){
	is.get();
	for(int i = 0; i < 2; i++){
		char ch = is.peek();
		if(ch == type){
			is.unget();
			return true;
		}
	}
	is.unget();
	return false;
}

void ReadToken(std::istream &is, std::string &token){

	is >> token;

  	if (is.fail()) {
    std::cout << "ReadToken, failed to read token at file position "
       << is.tellg();
    }

    if (!isspace(is.peek())) {
          std::cout << "ReadToken, expected space after token, saw instead "
              << static_cast<char>(is.peek())
              << ", at file position " << is.tellg();
  	}
  	
  	is.get();  // consume the space.

}
	
void ReadBasicType(std::istream &is, float* value ){

	is >> *value ;
}

void ReadBasicType(std::istream &is, int* value){

	is >> *value ;
}

void WriteToken(std::ostream &os, std::string& token){

  	os << token << " ";
  	if (os.fail()) {
    	throw std::runtime_error("Write failure in WriteToken.");
  	}
}

void WriteToken(std::ostream &os, const char* token){

	for(int i = 0 ; i < strlen(token); i++)
  		os << *(token+i) ;
  	os<<" " ;
  	if (os.fail()) {
    	throw std::runtime_error("Write failure in WriteToken.");
  	}
}

void WriteToken(std::ostream &os, const int& value){

	os << value << " ";

	if (os.fail()) {
    	throw std::runtime_error("Write failure in WriteToken.");
  	}

}

void WriteBasicType(std::ostream &os, const float& value){

	os << value << " ";
}

void WriteBasicType(std::ostream &os, const int& value){
	os << value << " ";
}


}
