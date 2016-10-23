// nnet/nnet-nnet.h
// Copyright 2014-11-14   (Author: xutao)
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
#ifndef NNET_NNET_H_
#define NNET_NNET_H_

#include <vector> 
#include <iostream>
#include <cstring>
#include <fstream>
#include "../cumatrix/cuMatrix.h"
#include "../base/common.h"
#include "../cumatrix/cuVector.h"
#include "../matrix/Matrix.h"
#include "../base/nnet-io.h"
#include "nnet-component.h"
#include "nnet-affinetransform.h"
#include "nnet-loss.h"
#include "nnet-train-options.h"

namespace nnet{
class Nnet
{
public:
	Nnet(){}
	~Nnet();
	//check the network
	void Check() const;
	//Forward of the network
	void ForwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat>* output);
	//Backword of the network
	void BackwardPropegation(cuMatrix<BaseFloat> &output_diff, cuMatrix<BaseFloat>* input_diff);
	// used for predicting
	void Feedforward( cuMatrix<BaseFloat> &in, cuMatrix<BaseFloat> *out) ;
	//return the input dimention
	int32 InputDim() const;
	//return the output dimention
	int32 OutputDim() const;
	//return the number of component of Net
	int32 NumComponent() const;
	//read net model from nnet model
	void Read(std::string &nnet_file) ;
	//write net model into a file
	void Write(std::string &file);
	//add Component type into the net
	void AppendComponent(Component* net_type);
	//initial with configure file
	void Init(const std::string &file);
	//initial the training options
	void SetTrnOption(NnetTrainOptions &opt) ;
	//release
	void Destroy();
	// training the network
	void Train(float* train_set, float* train_label, int32 num_train_samples, std::string objective_function);
	//cross-validation
	void CrossValidate(float* dev_set, float* dev_label, int32 num_dev_samples, std::string objective_function);
	// testing the network
	void Predict(float* test_set, float* test_label, int32 num_test_samples);

private:
	std::vector<Component*> Components_;
  	std::vector<cuMatrix<BaseFloat> > propagate_buf_; ///< buffers for forward pass
  	std::vector<cuMatrix<BaseFloat> > backpropagate_buf_; ///< buffers for backward pass
  	NnetTrainOptions TrnOptions_ ;
	/* data */
};


}//namespace nnet


#endif //NNET_NNET_H_
