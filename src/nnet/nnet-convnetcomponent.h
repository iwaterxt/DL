//nnet-convnetcomponent.h
// Copyright 2015-9-27   (Author: xutao)
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
#ifndef NNET_NNET_CONVNETCOMPONENT_H_
#define NNET_NNET_CONVNETCOMPONENT_H_

#include "nnet-component.h"

namespace nnet{
class ConvnetComponent : public UpdateComponent{
public:
	ConvnetComponent(int32 dim_in, int32 dim_out):UpdataComponent(dim_in, dim_out),
	Filter_()
	{}
	~ConvnetComponent(){}

	void Init();

	void ForwardPropegate(){

	}

	void BackwardPropegate(){

	}

	void Updata(){

	}
private:
	/*data*/
	int32 filter_map_x_, filter_map_y_, filter_len_x_, filter_len_y_,filter_step_x_, filter_step_y_;
	cuMatrix<BaseFloat> Filter_ ;
	cuVector<BaseFloat> Bias_ ;

	cuMatrix<BaseFloat> Filter_updata_;
	cuVector<BaseFloat> Bias_updata_;

};

}//end of namespace

#endif