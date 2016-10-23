//nnet-cudnn-2d-pooling.h
// Copyright 2016-9-08   (Author: xutao)
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

#ifndef NNET_CUDNN_2D_POOLING_H_
#define NNET_CUDNN_2D_POOLING_H_

#include <cudnn.h>
#include "nnet-component.h"
#include "../cumatrix/cuda-utils.h"

namespace nnet{
	
	struct PoolingParam{
		int fmap_input_w ;
		int fmap_input_h ;
		int fmap_output_w ;
		int fmap_output_h ;
		int pool_w ;
		int pool_h ;
		int stride_w ;
		int stride_h ;
		int pad_w ;
		int pad_h ;
		int num_filter ;
		int batch_size ;
	};

	class CudnnMaxPooling2DComponent : public Component{
	public:

		CudnnMaxPooling2DComponent(int32 dim_in, int32 dim_out):Component(dim_in, dim_out),mode_(CUDNN_POOLING_MAX),
		dtype_(CUDNN_DATA_FLOAT), Init_(false)
		{}
		~CudnnMaxPooling2DComponent()
		{
            if(Init_){
			    CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      		    CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      		    CHECK_EQ(cudnnDestroyPoolingDescriptor(pooling_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      		    cudaStreamDestroy(stream_);
      		    cudnnDestroy(handle_);
            }
		}

		ComponentType GetType() const{
	 		return kCudnnMaxPooling2DComponent;
		}

		bool IsUpdatable(){
	 	
	 		return false;
		}

    	Component* Copy() const { return new CudnnMaxPooling2DComponent(*this); }

    	void InitData(std::istream &is){

    		std::string token;
    		while(!is.eof()){
				ReadToken(is, token);
				 if(token == "<FmapXLen>") ReadBasicType(is, &param_.fmap_input_w);
				else if(token == "<FmapYLen>") ReadBasicType(is, &param_.fmap_input_h);
				else if(token == "<PoolXLen>") ReadBasicType(is, &param_.pool_w);
				else if(token == "<PoolYLen>") ReadBasicType(is, &param_.pool_h);
				else if(token == "<PoolXStep>") ReadBasicType(is, &param_.stride_h);
				else if(token == "<PoolYStep>") ReadBasicType(is, &param_.stride_w);
				else if(token == "<PoolXPad>") ReadBasicType(is, &param_.pad_w);
				else if(token == "<PoolYPad>") ReadBasicType(is, &param_.pad_h);
				else std::cout<<"Unknown Token "<<token << ", a typo in config? "
							  << "(FmapXLen|FmapYLen|PoolXStep|PoolYStep|PoolXLen|PoolYLen|PoolXPad|PoolYPad)";
				is >> std::ws ; 
			}
            assert(dim_in_ % (param_.fmap_input_h * param_.fmap_input_w)==0) ;
            param_.num_filter = dim_in_ /(param_.fmap_input_w * param_.fmap_input_h);


    	}

        void Init(){

            cudaStreamCreate(&stream_);
            cudnnCreate(&handle_);
            cudnnSetStream(handle_, stream_);
            param_.batch_size = opts_.minibatch_size ;
            nan_prop_ = CUDNN_NOT_PROPAGATE_NAN ;
            CHECK_EQ(cudnnCreatePoolingDescriptor(&pooling_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
            CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
            CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

            CHECK_EQ(cudnnSetTensor4dDescriptor(in_desc_,
                                                CUDNN_TENSOR_NCHW,
                                                dtype_,
                                                param_.batch_size,
                                                param_.num_filter,
                                                param_.fmap_input_h,
                                                param_.fmap_input_w), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

            CHECK_EQ(cudnnSetTensor4dDescriptor(out_desc_,
                                                CUDNN_TENSOR_NCHW,
                                                dtype_,
                                                param_.batch_size,
                                                param_.num_filter,
                                                param_.fmap_output_h,
                                                param_.fmap_output_w), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);


            CHECK_EQ(cudnnSetPooling2dDescriptor(pooling_desc_,
                                                 CUDNN_POOLING_MAX,
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 param_.pool_h,
                                                 param_.pool_w,
                                                 param_.pad_h,
                                                 param_.pad_w,
                                                 param_.stride_h,
                                                 param_.stride_w), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);



        }

    	void ReadData(std::istream &is){

    		ExpectToken(is, "<FmapXLen>");
    		ReadBasicType(is, &param_.fmap_input_w);
    		ExpectToken(is, "<FmapYLen>");
    		ReadBasicType(is, &param_.fmap_input_h);
    		ExpectToken(is, "<PoolXLen>");
    		ReadBasicType(is, &param_.pool_w);
    		ExpectToken(is, "<PoolYLen>");
    		ReadBasicType(is, &param_.pool_h);
    		ExpectToken(is, "<PoolXStep>");
    		ReadBasicType(is, &param_.stride_w);
    		ExpectToken(is, "<PoolYStep>");
    		ReadBasicType(is, &param_.stride_h);
    		ExpectToken(is, "<PoolXPad>");
    		ReadBasicType(is, &param_.pad_w);
    		ExpectToken(is, "<PoolYPad>");
    		ReadBasicType(is, &param_.pad_h);

            assert((param_.fmap_input_h + 2*param_.pad_h - param_.pool_h)%param_.stride_h == 0);
            assert((param_.fmap_input_w + 2*param_.pad_w - param_.pool_w)%param_.stride_w == 0);

            param_.fmap_output_h = (param_.fmap_input_h + 2*param_.pad_h - param_.pool_h)/param_.stride_h + 1 ;
            param_.fmap_output_w = (param_.fmap_input_w + 2*param_.pad_w - param_.pool_w)/param_.stride_w + 1 ;
            assert(dim_in_ % (param_.fmap_input_h * param_.fmap_input_w)==0) ;
            param_.num_filter = dim_in_ /(param_.fmap_input_w * param_.fmap_input_h);

    	}

    	void WriteData(std::ostream &os) {
    		WriteToken(os, "<FmapXLen>") ;
    		WriteBasicType(os, param_.fmap_input_w);
    		WriteToken(os, "<FmapYLen>") ;
    		WriteBasicType(os, param_.fmap_input_h);
    		WriteToken(os, "<PoolXLen>") ;
    		WriteBasicType(os, param_.pool_w);
    		WriteToken(os, "<PoolYLen>") ;
    		WriteBasicType(os, param_.pool_h);
    		WriteToken(os, "<PoolXStep>");
    		WriteBasicType(os, param_.stride_w);
    		WriteToken(os, "<PoolYStep>");
    		WriteBasicType(os, param_.stride_h);
    		WriteToken(os, "<PoolXPad>") ;
    		WriteBasicType(os, param_.pad_w);
    		WriteToken(os, "<PoolYPad>") ;
    		WriteBasicType(os, param_.pad_h);
            os << "\n";
    	}
    

    	void ForwardPropegation(cuMatrix<BaseFloat> &input , cuMatrix<BaseFloat> *output){

            if(!Init_){
                Init() ;
                Init_ = true;
            }

    		BaseFloat alpha = 1.0f, beta = 0.0f ;
    		BaseFloat* data_ptr = input.Data() ;
    		void* out_ptr = (void*)output->Data() ;
    		CHECK_EQ(cudnnPoolingForward(handle_,
    									 pooling_desc_,
    									 &alpha,
    									 in_desc_,
    									 data_ptr,
    									 &beta,
    									 out_desc_,
    									 out_ptr), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

    	}

    	void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff){

    		BaseFloat alpha = 1.0f, beta = 0.0f ;
    		BaseFloat* out_dptr = output.Data();
    		BaseFloat* g_out_dptr = output_diff.Data();
    		BaseFloat* in_dptr = input.Data();
    		BaseFloat* g_in_dptr = input_diff->Data();
    		CHECK_EQ(cudnnPoolingBackward(handle_,
    							 		  pooling_desc_,
    							 		  &alpha,
    							 		  out_desc_,
    							 		  out_dptr,
    							 		  out_desc_,
    							 		  g_out_dptr,
    							 		  in_desc_,
    							 		  in_dptr,
    							 		  &beta,
    							 		  in_desc_,
    							 		  g_in_dptr), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    	}

    	private:
    		cudnnDataType_t dtype_;
  			cudnnHandle_t handle_;
  			cudaStream_t stream_ ;
  			cudnnPoolingMode_t mode_;
  			cudnnTensorDescriptor_t in_desc_;
  			cudnnTensorDescriptor_t out_desc_;
  			cudnnPoolingDescriptor_t pooling_desc_;
  			cudnnNanPropagation_t nan_prop_;
  			PoolingParam param_;
            bool Init_ ;
	};


}

#endif