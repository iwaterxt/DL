// demo/smart-network-testing
// Copyright 2015-4-28 (Author: xutao)
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
#include "../base/nnet-io.h"
#include "../base/option-parse.h"

template <class Type>
static void stox(const std::string& str , Type& val)
{    
    val = atof(str.c_str()) ;
}

template <class Type>
static void LoadData(const char* file , Type* data)
{
    if(NULL == data)
        std::cout<<"not allocated pointer";

    Type* data_ptr = data ;
    std::string str , str1 ;

    std::ifstream ifs(file) ;
    if (!ifs)
    {
        std::cerr << "Error : cound not to open file : " << file << std::endl ;
        return ;
    }
 
    for (;std::getline(ifs , str , '\n');)
    {
        std::stringstream str_stream(str.c_str()) ;

        while(str_stream >> str1)
        {
            stox(str1 , *data_ptr );

            data_ptr++ ;
        }
    }

    ifs.close() ;
}

int main(int argc,  char* argv[])
{
    using namespace nnet ;
    Nnet net;

//=========2015-9-19================
    int32 minibatch_size = 0 , test_number = 0 , image_size = 0 ;
    bool apply_norm = 0 ;
    std::string TestFile , test_label , mlp_best ;
    OptionParse Parser;
    Parser.Register_int(argc, argv, "minibatch-size", minibatch_size, "the batch-size of training");
    Parser.Register_int(argc, argv, "test-number", test_number, "the number of test data");
    Parser.Register_int(argc, argv, "image-size", image_size, "the size of image");
    Parser.Register_bool(argc, argv, "apply-norm", apply_norm, "apply normalization");
    Parser.Register_string(argc, argv, "TestFile", TestFile, "the path of testing dataset");
    Parser.Register_string(argc, argv, "test-label", test_label, "the path of testing dataset");
    Parser.Register_string(argc, argv, "mlp-best", mlp_best, "the path of mlp-best model");
//==================================

    clock_t start,finish;
    double totaltime;
    start=clock();
    float* test_set   = new float[test_number*image_size];
    float* Test_label = new float[test_number];
    NnetTrainOptions opt ;
    opt.minibatch_size = minibatch_size ;
    opt.image_size = image_size ;
    opt.test_number = test_number ;
    opt.apply_norm = apply_norm ;
    

    //fill data
    std::cout<<">>>>First loading data<<<<<"<<std::endl;
    LoadData(TestFile.c_str(), test_set);
    LoadData(test_label.c_str(), Test_label);
    
    net.Read(mlp_best);
    net.SetTrnOption(opt) ;
    std::cout<<">>>>Begin Predicting<<<<"<<std::endl;
    net.Predict(test_set, Test_label, test_number);
    
    finish = clock();
    totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout<<"\n the running time is: "<<totaltime<<"seconds!"<<std::endl;
   delete []test_set;
   test_set = NULL;
   delete []Test_label;
   Test_label = NULL;
   return 0;
}
