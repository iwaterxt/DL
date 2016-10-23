// demo/smart-network-initial.cc
// Copyright 2015-9-19  (Author: xutao)
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

#include <sstream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "../nnet/nnet-nnet.h"
#include "../base/option-parse.h"

int main(int argc,  char* argv[]){

	using namespace nnet ;
	std::string prototype ;
	std::string nnet_initial ;
	OptionParse Parser;
	Parser.Register_string(argc, argv, "prototype", prototype, "the prototype of network model");
	Parser.Register_string(argc, argv, "nnet-initial", nnet_initial, "the prototype of network model");

	Nnet net;

	net.Init(prototype) ;

	net.Write(nnet_initial) ;

	return 0 ;
}