// nnet/nnet-nnet.h
// Copyright 2015-9-3  (Author: xutao)
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
#ifndef NNET_TRAIN_OPTION_H_
#define NNET_TRAIN_OPTION_H_

#include "../base/common.h"

namespace nnet{

struct NnetTrainOptions
{

  BaseFloat learn_rate ;
  BaseFloat bias_learn_rate ;
  BaseFloat momentum ;
  BaseFloat l2_penalty ;
  BaseFloat l1_penalty ;
  int32 minibatch_size ;
  int32	image_size ;
  int32 tr_number ;
  int32 cv_number ;
  int32 test_number ;
  int32 class_number ;
  bool apply_norm ;
  bool is_train ;
  

  NnetTrainOptions(): learn_rate(0.008),bias_learn_rate(0.0008),momentum(0.0),l2_penalty(0.0),l1_penalty(0.0), minibatch_size(0),
  image_size(0), tr_number(0), cv_number(0), test_number(0), class_number(0), apply_norm(0), is_train(0){}

  
};

} 


#endif



