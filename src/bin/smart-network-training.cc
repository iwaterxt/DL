// demo/smart-network-training
// Copyright 2015-4-28  (Author: xutao)
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
//========2015-9-19=============
    int32 minibatch_size = 0 , cv_number = 0 , tr_number = 0 , image_size = 0 , class_number = 0 ;
    bool apply_norm = 0 , cross_validate = 0 ;
    BaseFloat learn_rate = 0.0 , bias_learnrate = 0.0 , momentum = 0.0 , l1_penalty = 0.0 , l2_penalty = 0.0 ;
    std::string TrainFile , CrossValFile , tr_lable , cv_lable , mlp_best , mlp_next;
    OptionParse Parser;
    Parser.Register_int(argc, argv, "minibatch-size", minibatch_size, "the batch-size of training");
    Parser.Register_int(argc, argv, "cv-number", cv_number, "the cross validation number");
    Parser.Register_int(argc, argv, "tr-number", tr_number, "the training number");
    Parser.Register_int(argc, argv, "image-size", image_size, "the size of image");
    Parser.Register_int(argc, argv, "class-number", class_number, "the number of class");
    Parser.Register_bool(argc, argv, "apply-norm", apply_norm, "apply normalization");
    Parser.Register_bool(argc, argv, "cross-validate", cross_validate, "apply cross_validate");
    Parser.Register_float(argc, argv, "learn-rate", learn_rate, "learn_rate value of training");
    Parser.Register_float(argc, argv, "bias-learnrate", bias_learnrate, "bias_learnrate value of training");
    Parser.Register_float(argc, argv, "momentum", momentum, "momentum of training");
    Parser.Register_float(argc, argv, "l1-penalty", l1_penalty, "l1-penalty of training");
    Parser.Register_float(argc, argv, "l2-penalty", l2_penalty, "l2-penalty of training");
    Parser.Register_string(argc, argv, "TrainFile", TrainFile, "the path of training dataset");
    Parser.Register_string(argc, argv, "tr-lable", tr_lable, "the path of training dataset label");
    Parser.Register_string(argc, argv, "CrossValFile", CrossValFile, "the path of cross validation dataset ");
    Parser.Register_string(argc, argv, "cv-lable", cv_lable, "the path of cross validation dataset label");
    Parser.Register_string(argc, argv, "mlp-best", mlp_best, "the path of mlp-best model");
    Parser.Register_string(argc, argv, "mlp-next", mlp_next, "the path of mlp-next model");
//==============================
    clock_t start,finish;
    double totaltime;
    start=clock();
    Nnet net;
    net.Read(mlp_best);
    NnetTrainOptions opt ;
    opt.bias_learn_rate = bias_learnrate ;
    opt.learn_rate = learn_rate ;
    opt.minibatch_size = minibatch_size ;
    opt.momentum = momentum ;
    opt.l1_penalty = l1_penalty ;
    opt.l2_penalty = l2_penalty ;
    opt.image_size = image_size ;
    opt.tr_number = tr_number ;
    opt.cv_number = cv_number ;
    opt.class_number = class_number ;
    opt.apply_norm = apply_norm ;
    net.SetTrnOption(opt) ;
    if(cross_validate){
          float* dev_set     = new float[cv_number*image_size];
          float* dev_label   = new float[cv_number];
          std::cout<<">>>>First loading crossvalidation data<<<<<"<<std::endl;
          LoadData(CrossValFile.c_str(), dev_set);
          LoadData(cv_lable.c_str(), dev_label);
          net.CrossValidate(dev_set, dev_label, cv_number, "Xent");
          delete []dev_set;
          dev_set = NULL;
          delete []dev_label;
          dev_label = NULL;
    }else{
          float* train_set   = new float[tr_number*image_size];
          float* train_label = new float[tr_number];
          std::cout<<">>>>First loading training data<<<<<"<<std::endl;
          LoadData(TrainFile.c_str(), train_set);
          LoadData(tr_lable.c_str(), train_label);
          std::cout<<"...Begin training dnn network..."<<std::endl;
          net.Train(train_set, train_label, tr_number, "Xent");
          net.Write(mlp_next);
          delete []train_set;
          train_set = NULL;
          delete []train_label;
          train_label = NULL;
    }

    finish = clock();
    totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
    std::cout<<"the running time is: "<<totaltime<<"seconds!"<<std::endl;
   return 0;
}
