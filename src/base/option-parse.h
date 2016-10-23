//base/nnet-math.h
// Copyright 2015-9-4   (Author: xutao)
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
#ifndef OPTION_PARSE_H_
#define OPTION_PARSE_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <cstring>

class OptionParse
{
public:
	OptionParse()
	{}

	~OptionParse()
	{}

	void Register_int(int argc, char** ptr, char* option, int& value, char* describe);

	void Register_float(int argc, char** ptr, char* option, float& value, char* describe);

	void Register_string(int argc, char** ptr, char* option, std::string& path, char* describe);

	void Register_bool(int argc, char** ptr, char* option, bool& value, char* describe);

};

void OptionParse::Register_int(int argc, char** ptr, char* option, int& value, char* describe){
	
	for(int i = 1; i < argc; i++){

		int length = strlen(option);
		char str[length+1] ;
		bool match = 1;
		if(strlen(ptr[i])>length){
			for(int j = 0; j < length; j++)
				str[j] = ptr[i][j+2] ;
			str[length] = '\0';
			match = strcmp(option,str);
		}
		if(match == 0){
			printf("%s\n", ptr[i]);
			match = 1 ;
			char* p = &ptr[i][length+3] ;
			value = atoi(p);
		}
	}
	
}

void OptionParse::Register_float(int argc, char** ptr, char* option, float& value, char* describe){

	for(int i = 1; i < argc; i++){

		int length = strlen(option);
		char str[length+1] ;
		bool match = 1;
		if(strlen(ptr[i])>length){
			for(int j = 0; j < length; j++)
				str[j] = ptr[i][j+2] ;
			str[length] = '\0';
			
			match = strcmp(option,str);
		}

		if(match == 0){
			printf("%s\n", ptr[i]);
			match = 1 ;
			char* p = &ptr[i][length+3] ;
			value = atof(p);
		}
	}

}

void OptionParse::Register_string(int argc, char** ptr, char* option, std::string& path, char* describe){

	char* p = NULL ;
	for(int i = 1; i < argc; i++){

		int length = strlen(option);
		char str[length+1] ;
		bool match = 1;
		if(strlen(ptr[i])>length){

			for(int j = 0; j < length; j++)
				str[j] = ptr[i][j+2] ;
			str[length] = '\0';
			match = strcmp(option,str);
		}

		if(match == 0){
			printf("%s\n", ptr[i]);
			match = 1 ;
			p = &ptr[i][length+3] ;
			std::string s(p);
			path = s ;
		}
	}
}

void OptionParse::Register_bool(int argc, char** ptr, char* option, bool& value, char* describe){

	for(int i = 1; i < argc; i++){

		int length = strlen(option);
		char str[length+1] ;
		bool match = 1;
		if(strlen(ptr[i])>length){
			for(int j = 0; j < length; j++)
				str[j] = ptr[i][j+2] ;
			str[length] = '\0';
			match = strcmp(option,str);
		}

		if(match == 0){
			printf("%s\n", ptr[i]);
			match = 1 ;
			char* p = &ptr[i][length+3] ;
			value = atoi(p);
		}
	}
}


#endif