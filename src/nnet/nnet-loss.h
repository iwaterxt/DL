// nnet/nnet-loss.h
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

#ifndef NNET_NNET_LOSS_H_
#define NNET_NNET_LOSS_H_

# include "../cumatrix/cuMatrix.h"
# include "../matrix/Matrix.h"
# include "../base/common.h"
# include <stdio.h>

namespace nnet{

class Xent{
public:
	Xent():cross_entropy_(0),number_(0)
	{}
	~Xent() {}

	void Eval(cuMatrix<BaseFloat> &net_out, cuMatrix<BaseFloat> &targets, cuMatrix<BaseFloat> *diff);

	void Report();


private:
	BaseFloat cross_entropy_;

	int32 number_;
};

class Mse{
public:
	Mse():mse_(0), mse_inc_(0), number_(0)
	{}
	~Mse(){}

	void Eval(cuMatrix<BaseFloat> &net_out, cuMatrix<BaseFloat> &targets, cuMatrix<BaseFloat> *diff);

	void Report();

private:
	BaseFloat mse_;

	BaseFloat mse_inc_;

	int32 number_;

};

}//end of namespace

#endif
