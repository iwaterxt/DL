//nnet-cudnn-active.h
// Copyright 2016-9-14   (Author: xutao)
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

#include <cudnn.h>
#include "nnet-component.h"
#include "../cumatrix/cuda-utils.h"

namespace nnet{

	struct ActiveParam{
		int num_filter ;
		int batch_size ;
	};

	class ActiveFunction: public Component{
	public:

		ActiveFunction
		(int32 dim_in, int32 dim_out):Component(dim_in, dim_out),init_cudnn_(false),ceil_(0.0)
		{}

		~ActiveFunction(){

			if(init_cudnn_){

				CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__) ;
				CHECK_EQ(cudnnDestroyActivationDescriptor(desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

			}

      		cudaStreamDestroy(stream_);
      		cudnnDestroy(handle_);
		}

		ComponentType GetType() const{
	 		return kActiveFunction;
		}

		bool IsUpdatable(){
	 	
	 		return false;
		}

		Component* Copy() const { return new ActiveFunction(*this); }

    	void InitData(std::istream &is){

    		std::string token;
    		while(!is.eof()){
				ReadToken(is, token);
				/**/ if(token == "<FunctionSign>") ReadBasicType(is, &sign_);
				else std::cout<<"Unknown Token "<<token << ", a typo in config? "
							  << "(FunctionSign)";
				is >> std::ws ; 

			}
    	}

    	void Init(int32 dim){

    		switch ((int)sign_){

    			case 1:
    				mode_ = CUDNN_ACTIVATION_RELU ;
    				break;
    			case 2:
    				mode_ = CUDNN_ACTIVATION_SIGMOID ;
    				break;
    			case 3:
    				mode_ = CUDNN_ACTIVATION_TANH ;
    				break;

    			default:
    				std::cout<<"Not implmented" ;
    				break;
    		}

    		init_cudnn_ = true ;

    		param_.batch_size = opts_.minibatch_size ;
    		param_.num_filter = dim ;
            
    		cudaStreamCreate(&stream_);
            cudnnCreate(&handle_);
            cudnnSetStream(handle_, stream_);

    		CHECK_EQ(cudnnCreateActivationDescriptor(&desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    		CHECK_EQ(cudnnSetActivationDescriptor(desc_, mode_, CUDNN_NOT_PROPAGATE_NAN, ceil_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    		CHECK_EQ(cudnnCreateTensorDescriptor(&shape_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    		CHECK_EQ(cudnnSetTensor4dDescriptor(shape_desc_,
    											CUDNN_TENSOR_NCHW,
    											CUDNN_DATA_FLOAT,
    											param_.batch_size,
    											param_.num_filter,
    											1,
    											1), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    	}

    	void ReadData(std::istream &is){
    		ExpectToken(is, "<FunctionSign>");
    		ReadBasicType(is, &sign_);
    	}

    	void WriteData(std::ostream &os) {

    		WriteToken(os, "<FunctionSign>") ;
    		WriteBasicType(os, sign_);
    	}

    	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){

    		if(!init_cudnn_){

    			Init(input.NumCols());
    		}
    	

    		BaseFloat alpha = 1.0f ;
    		BaseFloat beta = 0.0f ;
    		const BaseFloat* data_ptr = input.Data() ;
    		BaseFloat* out_ptr = output->Data() ;


    		CHECK_EQ(cudnnActivationForward(handle_,
    										desc_,
    										&alpha,
    										shape_desc_,
    										data_ptr,
    										&beta,
    										shape_desc_,
    										out_ptr), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

		}

		void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){


			BaseFloat alpha = 1.0f ;
			BaseFloat beta = 0.0f ;
			const BaseFloat* out_ptr = output.Data() ;
			const BaseFloat* data_ptr = input.Data() ;
			BaseFloat* input_grad_ptr = input_diff->Data() ;
			const BaseFloat* grad_ptr = output_diff.Data() ;

			CHECK_EQ(cudnnActivationBackward(handle_,
											 desc_,
											 &alpha,
											 shape_desc_,
											 out_ptr,
											 shape_desc_,
											 grad_ptr,
											 shape_desc_,
											 data_ptr,
											 &beta,
											 shape_desc_,
											 input_grad_ptr), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

		}
	private:
		int32 sign_ ;
		bool init_cudnn_ ;
		cudnnActivationMode_t mode_ ;
		cudnnTensorDescriptor_t shape_desc_ ;
		cudnnActivationDescriptor_t desc_ ;
		double ceil_ ;
		cudnnHandle_t handle_;
  		cudaStream_t stream_ ;
  		ActiveParam param_ ;

};//end of class


}//end of namespace