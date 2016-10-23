//nnet/nnet-loss.cc
// Copyright 2014-12-31  (Author: xutao)
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

#include "../nnet/nnet-loss.h"
#include <math.h>

namespace nnet{

  void Xent::Eval(cuMatrix<BaseFloat> &net_out, cuMatrix<BaseFloat> &target, cuMatrix<BaseFloat> *diff){


      diff->CopyFromMat(net_out);

      diff->AddMat(target, kNoTrans, -1.0);

      cuMatrix<BaseFloat> Entropy(net_out.NumRows(), net_out.NumCols());

      Entropy.CopyFromMat(net_out);

      Entropy.Log();

      Entropy.MulElements(target);

      cross_entropy_ = - Entropy.Sum();

      Report();
   }

  void  Xent::Report(){

      number_++;
      if(number_%2 == 0){

         printf("LOG：after %d epochs of training, the Xent is %f \n", number_/2, cross_entropy_);
         cross_entropy_ = 0;
      }

   }
   
  void  Mse::Eval(cuMatrix<BaseFloat> &net_out, cuMatrix<BaseFloat> &target, cuMatrix<BaseFloat> *diff){

   	assert(net_out.NumRows()==target.NumRows()&&net_out.NumCols()==target.NumCols());

      diff->Resize(net_out.NumRows(), net_out.NumCols()) ;

      diff->CopyFromMat(net_out);

      diff->AddMat(target, kNoTrans, -1.0);

      cuMatrix<BaseFloat> diff_copy(net_out.NumRows(), net_out.NumCols());

      diff_copy.CopyFromMat(*diff);

      diff_copy.MulElements(diff_copy);

   	mse_inc_ = diff_copy.Sum();

   	mse_ = sqrt(mse_inc_);
   }

  void Mse::Report(){

   	number_++;
   	if(number_%2 == 0){

   		printf("LOG：after %d epochs of training, the Mse is %f ", number_/2, mse_);
   	      mse_ = 0;
      }
   }
}//end of namespace 