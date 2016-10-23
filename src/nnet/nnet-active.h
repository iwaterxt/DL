//nnet-active.h
// Copyright 2014-12-29   (Author: xutao)
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

#ifndef NNET_NNET_ACTIVE_H_
#define NNET_NNET_ACTIVE_H_

#include <cudnn.h>
#include "nnet-component.h"
#include "../cumatrix/cuda-utils.h"

//ey : output_diff
//y  : out
//ex : in_diff
//x  : in
 
namespace nnet{

class SoftMax : public Component{

public:
	SoftMax(int32 dim_in, int32 dim_out):Component(dim_in, dim_out)
	{}
	~SoftMax()
	{}
	ComponentType GetType() const{
	 	return kSoftMax;
	}

	bool IsUpdatable(){
	 	
	 	return false;
	}

    Component* Copy() const { return new SoftMax(*this); }
    
	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){
		output->CopyFromMat(input);
		output->ApplySoftMaxPerRow();
	}

	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){
		input_diff->CopyFromMat(output_diff);
	}

};

class CudnnSoftMax : public Component{

public:
	CudnnSoftMax(int32 dim_in, int32 dim_out):Component(dim_in, dim_out), dtype_(CUDNN_DATA_FLOAT),init_cudnn_(false)
	{}
	~CudnnSoftMax(){

		if(init_cudnn_){
			CHECK_EQ(cudnnDestroyTensorDescriptor(shape_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
		}

      	cudaStreamDestroy(stream_);
      	cudnnDestroy(handle_);
	}

	bool IsUpdatable(){
		return false;
	}

	ComponentType GetType()const{
		return kCudnnSoftMax ;
	}

	Component* Copy() const { return new CudnnSoftMax(*this); }

	void Init(int32 dim){

		init_cudnn_ = true ;

    	cudaStreamCreate(&stream_);
        cudnnCreate(&handle_);
        cudnnSetStream(handle_, stream_);

        int32 batch_size = opts_.minibatch_size ;

		CHECK_EQ(cudnnCreateTensorDescriptor(&shape_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

		CHECK_EQ(cudnnSetTensor4dDescriptor(shape_desc_,
											CUDNN_TENSOR_NCHW,
											CUDNN_DATA_FLOAT,
											batch_size,
											dim,
											1,
											1), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

	}

	void ForwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat> *output){

		if(!init_cudnn_){

			Init(input.NumCols());

		}

		BaseFloat alpha = 1.0f ;
		BaseFloat beta = 0.0f ;
		const BaseFloat* data_ptr = input.Data() ;
		BaseFloat* out_ptr = output->Data() ;


		CHECK_EQ(cudnnSoftmaxForward(handle_,
									CUDNN_SOFTMAX_FAST,
									CUDNN_SOFTMAX_MODE_CHANNEL,
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
		const BaseFloat* out_ptr = output.Data();
		BaseFloat* in_grad_ptr = input_diff->Data();
		const BaseFloat* out_grad_ptr = output_diff.Data();

		CHECK_EQ(cudnnSoftmaxBackward(handle_,
									  CUDNN_SOFTMAX_FAST,
									  CUDNN_SOFTMAX_MODE_CHANNEL,
									  &alpha,
									  shape_desc_,
									  out_ptr,
									  shape_desc_,
									  out_grad_ptr,
									  &beta,
									  shape_desc_,
									  in_grad_ptr), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);


	}
private:
	bool init_cudnn_ ;
	cudnnTensorDescriptor_t shape_desc_ ;
	cudnnDataType_t dtype_ ;
	cudnnHandle_t handle_;
  	cudaStream_t stream_ ;
};

class Sigmoid : public Component{

public:

	Sigmoid(int32 dim_in, int32 dim_out):Component(dim_in, dim_out)
	{}
	
	~Sigmoid()
	{}

	ComponentType GetType() const{
	 	return kSigmoid;
	}

	bool IsUpdatable(){
	 	
	 	return false;
	}


	Component* Copy() const { return new Sigmoid(*this); }

	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){
		//y = 1/(1+e^(-x))

		output->Sigmoid(input);

	}

	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){
		//ex = x(1-x)ey

		input_diff->DiffSigmoid(output, output_diff);

	}

};

class Tanh : public Component{

public:
	Tanh(int32 dim_in, int32 dim_out):Component(dim_in, dim_out)
	{}
	~Tanh()
	{}

	ComponentType GetType() const {
	 	return kTanh;
	}

	bool IsUpdatable(){
	 	
	 	return false;
	}


	Component* Copy() const { return new Tanh(*this); }

	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){
		//y = (e^x - e^(-x))/(e^x + e^(-x))
		output->Tanh(input);
	}

	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){
		//ex = (1-y^2)ey
		input_diff->DiffTanh(output, output_diff);
	}
};

class ReLU : public Component{

public:
	ReLU(int32 dim_in, int32 dim_out):Component(dim_in, dim_out)
	{}
	~ReLU()
	{}

	ComponentType GetType() const{
	 	return kReLU;
	}

	bool IsUpdatable(){
	 	
	 	return false;
	}

	Component* Copy() const { return new ReLU(*this); }

	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){
		//y = {x>=0? x , 0}
		output->ReLU(input);
	}

	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){
		//ex = {ey>=0? ey, 0}
		input_diff->DiffReLU(output, output_diff);
	}

};


class DropOut : public Component{
public:
	DropOut(int32 dim_in, int32 dim_out):Component(dim_in, dim_out),DropOutRetention_(0.5)
	{}
	~DropOut()
	{}

	ComponentType GetType() const{
		
	 	return kDropOut;
	}

	bool IsUpdatable(){
	 	
	 	return false;
	}


	Component* Copy() const { return new DropOut(*this); }

	void Init(BaseFloat DropOutRetention){

		DropOutRetention_ = DropOutRetention;
	}
	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){

		Mask_.Resize(output->NumRows(), output->NumCols()) ;
		Mask_.BinarizeProbs(DropOutRetention_) ;
		output->CopyFromMat(input) ;
		output->MulElements(Mask_) ;
		output->Scale(1.0 / DropOutRetention_) ;
	}
	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){

		input_diff->CopyFromMat(output_diff) ;
		input_diff->MulElements(Mask_) ;
		input_diff->Scale(1.0 / DropOutRetention_) ;
	}

private:

	BaseFloat DropOutRetention_ ;
	cuMatrix<BaseFloat> Mask_ ;
};

}//namespace 

#endif
