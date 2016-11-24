//nnet-cudnn-convnetcomponent.h
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
#ifndef NNET_CUDNN_BATCH_NORM_H_
#define NNET_CUDNN_BATCH_NORM_H_

#include <cudnn.h>
#include "nnet-component.h"
#include "../cumatrix/cuda-utils.h"
#include "../base/nnet-math.h"

namespace nnet{

class BatchNormCudnn : public UpdateComponent{

public:
	BatchNormCudnn(int32 dim_in, int32 dim_out):UpdateComponent(dim_in, dim_out),num_filters_(0), fmapxlen_(0), fmapylen_(0){}

	~BatchNormCudnn(){
		if(initialize_){
      		CHECK_EQ(cudnnDestroyTensorDescriptor(io_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      		CHECK_EQ(cudnnDestroyTensorDescriptor(mean_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      		cudaStreamDestroy(stream_);
      		cudnnDestroy(handle_);
      	}
	}

	Component* Copy() const {return new BatchNormCudnn(*this);}

	ComponentType GetType() const{return kBatchNormCudnn;}

	void InitData(std::istream &is){

		float gamma_init = 1.0f, beta_init = 0.0f;
		std::string token ;
		while(!is.eof()){
			ReadToken(is, token);
			if(token == "<Gamma>") ReadBasicType(is, &gamma_init);
			else if(token == "<Beta>") ReadBasicType(is, &beta_init);
			else if(token == "<NumFilters>") ReadBasicType(is, &num_filters_);
			else if(token == "<FmapXLen>") ReadBasicType(is, &fmapxlen_);
			else if(token == "<FmapYLen>") ReadBasicType(is, &fmapylen_);
			else std::cout << "Unknown token " << token << ", a typo in config? "
			<<"Mean|Var|NumFilters|FmapYLen|FmapXLen";
		}

		gamma_.Resize(num_filters_);
		gamma_.Set(gamma_init);
		beta_.Resize(num_filters_);
		beta_.Set(beta_init);
		dgamma_.Resize(num_filters_);
		dbeta_.Resize(num_filters_);
		moving_mean_.Resize(num_filters_);
		moving_inv_var_.Resize(num_filters_);
		save_mean_.Resize(num_filters_);
		save_inv_var_.Resize(num_filters_);

	}

	void ReadData(std::istream &is){

		ExpectToken(is, "<NumFilters>");
		ReadBasicType(is, &num_filters_);
		ExpectToken(is, "<FmapXLen>");
		ReadBasicType(is, &fmapxlen_);
		ExpectToken(is, "<FmapYLen>");
		ReadBasicType(is, &fmapylen_);
		gamma_.Resize(num_filters_);
		beta_.Resize(num_filters_);

		ExpectToken(is, "<Gamma>");
		gamma_.Read(is);
		ExpectToken(is, "<Beta>");
		beta_.Read(is);

		dgamma_.Resize(num_filters_);
		dbeta_.Resize(num_filters_);
		moving_mean_.Resize(num_filters_);
		moving_inv_var_.Resize(num_filters_);
		save_mean_.Resize(num_filters_);
		save_inv_var_.Resize(num_filters_);


	}

	void WriteData(std::ostream &os){

  		WriteToken(os, "<NumFilters>");
  		WriteBasicType(os, num_filters_);
  		WriteToken(os, "<FmapXLen>");
  		WriteBasicType(os, fmapxlen_);
  		WriteToken(os, "<FmapYLen>");
  		WriteBasicType(os, fmapylen_);

		WriteToken(os, "<Gamma>");
		gamma_.Write(os);
		WriteToken(os, "<Beta>");
		beta_.Write(os);
	}

	void Init(int32 batch_size){

        cudaStreamCreate(&stream_);
        cudnnCreate(&handle_);
        cudnnSetStream(handle_, stream_);

		CHECK_EQ(cudnnCreateTensorDescriptor(&io_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
		CHECK_EQ(cudnnCreateTensorDescriptor(&mean_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
		CHECK_EQ(cudnnSetTensor4dDescriptor(io_desc_,
											CUDNN_TENSOR_NCHW,
											CUDNN_DATA_FLOAT,
											batch_size,
											num_filters_,
											fmapxlen_,
											fmapylen_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
		CHECK_EQ(cudnnSetTensor4dDescriptor(mean_desc_,
											CUDNN_TENSOR_NCHW,
											CUDNN_DATA_FLOAT,
											1,
											num_filters_,
											1,
											1), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
	}

	void ForwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *output){

		float a = 1.0f, b = 0.0f;
		if(!initialize_){
			Init(input.NumRows());
			initialize_ = true ;
		}
		BaseFloat momentum = opts_.momentum ;
		BaseFloat* in_dptr = input.Data();
		BaseFloat* out_dptr = output->Data();
		BaseFloat* gamma_dptr = gamma_.Data();
		BaseFloat* beta_dptr = beta_.Data();
		BaseFloat* moving_mean_dptr = moving_mean_.Data();
		BaseFloat* moving_inv_var_dptr = moving_inv_var_.Data();
		BaseFloat* save_mean_dptr = save_mean_.Data();
		BaseFloat* save_inv_var_dptr = save_inv_var_.Data();

		if(opts_.is_train){

			CHECK_EQ(cudnnBatchNormalizationForwardTraining(handle_,
															CUDNN_BATCHNORM_SPATIAL,
															&a,
															&b,
															io_desc_,
															in_dptr,
															io_desc_,
															out_dptr,
															mean_desc_,
															gamma_dptr,
															beta_dptr,
															1-momentum,
															moving_mean_dptr,
															moving_inv_var_dptr,
															1e-3f,
															save_mean_dptr,
															save_inv_var_dptr), CUDNN_STATUS_SUCCESS,__FILE__,__LINE__);

		}else{

			CHECK_EQ(cudnnBatchNormalizationForwardInference(handle_,
															 CUDNN_BATCHNORM_SPATIAL,
															 &a,
															 &b,
															 io_desc_,
															 in_dptr,
															 io_desc_,
															 out_dptr,
															 mean_desc_,
															 gamma_dptr,
															 beta_dptr,
															 moving_mean_dptr,
															 moving_inv_var_dptr,
															 1e-3f), CUDNN_STATUS_SUCCESS,__FILE__,__LINE__);
		}

	}

	void BackwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *input_diff, cuMatrix<BaseFloat> &output, cuMatrix<BaseFloat> &output_diff){
	
		float a = 1.0f, b = 0.0f;

		BaseFloat* in_dptr = input.Data();
		BaseFloat* din_dptr = input_diff->Data();
		BaseFloat* dout_dptr = output_diff.Data();
		BaseFloat* gamma_dptr = gamma_.Data();
		BaseFloat* dgamma_dptr = dgamma_.Data();
		BaseFloat* dbeta_dptr = dbeta_.Data();
		BaseFloat* save_mean_dptr = save_mean_.Data();
		BaseFloat* save_inv_var_dptr = save_inv_var_.Data();

		CHECK_EQ(cudnnBatchNormalizationBackward(handle_,
												 CUDNN_BATCHNORM_SPATIAL,
												 &a,
												 &b,
												 &a,
												 &b,
												 io_desc_,
												 in_dptr,
												 io_desc_,
												 dout_dptr,
												 io_desc_,
												 din_dptr,
												 mean_desc_,
												 gamma_dptr,
												 dgamma_dptr,
												 dbeta_dptr,
												 1e-3f,
												 save_mean_dptr,
												 save_inv_var_dptr), CUDNN_STATUS_SUCCESS,__FILE__, __LINE__);

	}

	void Update(cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &output_diff){

		BaseFloat lr = opts_.learn_rate ;
    	gamma_.AddVec(dgamma_, -lr);
    	beta_.AddVec(dbeta_, -lr);
	}

private:
	int32 num_filters_, fmapxlen_, fmapylen_;

	cudnnTensorDescriptor_t io_desc_;
	cudnnTensorDescriptor_t mean_desc_;

	cuVector<BaseFloat> gamma_ ;
	cuVector<BaseFloat> beta_ ;

	cuVector<BaseFloat> dgamma_ ;
	cuVector<BaseFloat> dbeta_ ;

	cuVector<BaseFloat> moving_mean_ ;
	cuVector<BaseFloat> moving_inv_var_ ;

	cuVector<BaseFloat> save_mean_;
	cuVector<BaseFloat> save_inv_var_ ;

	bool initialize_ ;
	cudnnHandle_t handle_ ;
  	cudaStream_t stream_ ;

};



}
#endif