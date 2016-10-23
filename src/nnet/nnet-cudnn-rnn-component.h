//nnet-cudnn-rnn-component.h
// Copyright 2016-9-20   (Author: xutao)
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

#ifndef NNET_CUDNN_RNN_COMPONENT_H_
#define NNET_CUDNN_RNN_COMPONENT_H_

namespace kaldi{
namespace nnet{

class CudnnRnnComponent : public UpdateComponent{

public:
	CudnnRnnComponent(int32 dim_in, int32 dim_out):UpdateComponent(dim_in, dim_out)
	{}

	~CudnnRnnComponent()
	{}

	Component* Copy() const {return new CudnnRnnComponent(*this);}

	ComponentType GetType() const {return kCudnnRnnComponent;}

	void InitData(std::istream &is){

		BaseFloat bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1 ;
		std::string token ;
		while(!is.eof()){
			ReadToken(is, token); 

		}

	}

	void Init(){


	}

	void ReadData(std::istream &is){

	}

	void WriteData(std::ostream &os) {

	}

	void ForwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *output){

	}

	void BackwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *input_diff, cuMatrix<BaseFloat> &output, cuMatrix<BaseFloat> &output_diff){

	}

	void Update(cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &output_diff){

	}

private:
	





};

}

}
#endif