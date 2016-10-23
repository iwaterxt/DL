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
#ifndef NNET_CUDNN_CONVNETCOMPONENT_H_
#define NNET_CUDNN_CONVNETCOMPONENT_H_

#include <cudnn.h>
#include "nnet-component.h"
#include "../cumatrix/cuda-utils.h"

namespace nnet{

struct ConvolutionParam {
	int num_output_filter;
	int num_input_filter;
	int fmap_output_h;
	int fmap_output_w;
	int fmap_input_h;
	int fmap_input_w;
	int kernel_h;
	int kernel_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
	int batch_size;
};

class ConvnetComponentCudnn : public UpdateComponent{

public:
	ConvnetComponentCudnn(int32 dim_in, int32 dim_out):UpdateComponent(dim_in, dim_out),Init_(false),forward_workspace_ptr_(NULL),
  backward_workspace_ptr_(NULL)
	{}

	~ConvnetComponentCudnn(){
    if(Init_){
      	CHECK_EQ(cudnnDestroyTensorDescriptor(in_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      	CHECK_EQ(cudnnDestroyTensorDescriptor(out_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      	CHECK_EQ(cudnnDestroyTensorDescriptor(bias_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      	CHECK_EQ(cudnnDestroyFilterDescriptor(filter_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
      	CHECK_EQ(cudnnDestroyConvolutionDescriptor(conv_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

      	cudaStreamDestroy(stream_);
      	cudnnDestroy(handle_);
      	cudaFree(forward_workspace_ptr_);
        cudaFree(backward_workspace_ptr_);
    }
	}

	Component* Copy() const {return new ConvnetComponentCudnn(*this);}

	ComponentType GetType() const{return kConvnetComponentCudnn;}

	void InitData(std::istream &is){

		BaseFloat bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1 ;
		std::string token ;
		while(!is.eof()){
			ReadToken(is, token); 
	    	/**/ if (token == "<ParamStddev>") ReadBasicType(is, &param_stddev);
      		else if (token == "<BiasMean>")    ReadBasicType(is, &bias_mean);
      		else if (token == "<BiasRange>")   ReadBasicType(is, &bias_range);
      		else if (token == "<FmapXLen>")    ReadBasicType(is, &param_.fmap_input_w);
      		else if (token == "<FmapYLen>")    ReadBasicType(is, &param_.fmap_input_h);
      		else if (token == "<FmapXPad>")	   ReadBasicType(is, &param_.pad_w);
      		else if (token == "<FmapYPad>")	   ReadBasicType(is, &param_.pad_h);
      		else if (token == "<FiltXLen>")    ReadBasicType(is, &param_.kernel_w);
      		else if (token == "<FiltYLen>")    ReadBasicType(is, &param_.kernel_h);
      		else if (token == "<FiltXStep>")   ReadBasicType(is, &param_.stride_w);
      		else if (token == "<FiltYStep>")   ReadBasicType(is, &param_.stride_h);
      		else if (token == "<LearnRateCoef>") ReadBasicType(is, &learn_rate_coef_);
      		else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, &bias_learn_rate_coef_);
      		else std::cout << "Unknown token " << token << ", a typo in config? "
                     	   << "(ParamStddev|BiasMean|BiasRange|FmapXLen|FmapYLen|FmapXPad|FmapYPad|"
                        	  "FiltXLen|FiltYLen|FiltXStep|FiltYStep|"
                              "LearnRateCoef|BiasLearnRateCoef)";
		}


    assert( (param_.fmap_input_h + 2*param_.pad_h - param_.kernel_h)%param_.stride_h == 0 );
    assert( (param_.fmap_input_w + 2*param_.pad_w - param_.kernel_w)%param_.stride_w == 0 );
    param_.fmap_output_h = (param_.fmap_input_h + 2*param_.pad_h - param_.kernel_h)/param_.stride_h + 1 ;
    param_.fmap_output_w = (param_.fmap_input_w + 2*param_.pad_w - param_.kernel_w)/param_.stride_w + 1 ;
    assert(dim_out_ % (param_.fmap_output_w * param_.fmap_output_h) == 0) ;
    assert(dim_in_ % (param_.fmap_input_w * param_.fmap_input_h) == 0) ;
    param_.num_output_filter = dim_out_ / (param_.fmap_output_w * param_.fmap_output_h) ;
    param_.num_input_filter = dim_in_ / (param_.fmap_input_w * param_.fmap_input_h) ;


		Matrix<BaseFloat> mat(param_.num_output_filter, param_.num_input_filter * param_.kernel_h * param_.kernel_w);
		for(int32 r = 0; r < param_.num_output_filter; r++)
			for(int32 c = 0; c < param_.num_input_filter * param_.kernel_w * param_.kernel_h; c++){
				mat(r,c) = param_stddev * RandGauss();
			}
    Filter_.Resize(param_.num_output_filter, param_.num_input_filter * param_.kernel_h * param_.kernel_w);
		Filter_ = mat ;

    Vector<BaseFloat> vec(param_.num_output_filter);
    for (int32 i=0; i<param_.num_output_filter; i++) {
      		// +/- 1/2*bias_range from bias_mean:
      	 	vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    Bias_.Resize(param_.num_output_filter);
    Bias_ = vec;
	}

  void Init(){
        //convert MB to Words
        //size_t workspace_byte = 8*1024*1024;
        size_t back_size = 0 ;
        size_t back_size_w = 0 ;
        cudaStreamCreate(&stream_);
        cudnnCreate(&handle_);
        cudnnSetStream(handle_, stream_);
        format_ = CUDNN_TENSOR_NCHW ;
        param_.batch_size = opts_.minibatch_size ;
        CHECK_EQ(cudnnCreateTensorDescriptor(&in_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnCreateTensorDescriptor(&out_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnCreateTensorDescriptor(&bias_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnCreateFilterDescriptor(&filter_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnCreateConvolutionDescriptor(&conv_desc_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnSetFilter4dDescriptor(filter_desc_,
                                            CUDNN_DATA_FLOAT,
                                            format_,
                                            param_.num_output_filter,
                                            param_.num_input_filter,
                                            param_.kernel_h,
                                            param_.kernel_w), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
        CHECK_EQ(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                param_.pad_h,
                                                param_.pad_w,
                                                param_.stride_h,
                                                param_.stride_w,
                                                1,
                                                1,
                                                CUDNN_CROSS_CORRELATION), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

        CHECK_EQ(cudnnSetTensor4dDescriptorEx(in_desc_,
                                              CUDNN_DATA_FLOAT,
                                              param_.batch_size,
                                              param_.num_input_filter,
                                              param_.fmap_input_h,
                                              param_.fmap_input_w,
                                              param_.fmap_input_h * param_.fmap_input_w * param_.num_input_filter,
                                              param_.fmap_input_h * param_.fmap_input_w,
                                              param_.fmap_input_w,
                                              1), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

        CHECK_EQ(cudnnSetTensor4dDescriptorEx(out_desc_,
                                              CUDNN_DATA_FLOAT,
                                              param_.batch_size,
                                              param_.num_output_filter,
                                              param_.fmap_output_h ,
                                              param_.fmap_output_w ,
                                              param_.fmap_output_h * param_.fmap_output_w * param_.num_output_filter,
                                              param_.fmap_output_h * param_.fmap_output_w,
                                              param_.fmap_output_w,
                                              1), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

        bias_offset_ = param_.num_output_filter;
        std::vector<int> bias_shape = {1, (int)bias_offset_, 1, 1};
        std::vector<int> bias_stride = {(int)bias_offset_, 1, 1, 1};
        CHECK_EQ(cudnnSetTensorNdDescriptor(bias_desc_,
                                            CUDNN_DATA_FLOAT,
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

      
        CHECK_EQ(cudnnGetConvolutionForwardAlgorithm(handle_,
                 in_desc_,
                 filter_desc_,
                 conv_desc_,
                 out_desc_,
                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                 0,
                 &algo_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

        CHECK_EQ(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
                 in_desc_,
                 out_desc_,
                 conv_desc_,
                 filter_desc_,
                 CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                 0,
                 &back_algo_w_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

        CHECK_EQ(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
                 filter_desc_,
                 out_desc_,
                 conv_desc_,
                 in_desc_,
                 CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                 0,
                 &back_algo_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);


      

        CHECK_EQ(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
                filter_desc_,
                out_desc_,
                conv_desc_,
                in_desc_,
                back_algo_,
                &back_size), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);


        CHECK_EQ(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
                in_desc_,
                out_desc_,
                conv_desc_,
                filter_desc_,
                back_algo_w_,
                &back_size_w), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);


        backward_workspace_byte_ = std::max(back_size, back_size_w);

        CHECK_EQ(cudnnGetConvolutionForwardWorkspaceSize(handle_,
                in_desc_,
                filter_desc_,
                conv_desc_,
                out_desc_,
                algo_,
                &forward_workspace_byte_), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__); 

        forward_workspace_ = forward_workspace_byte_ / sizeof(float) + 1;
        backward_workspace_ = backward_workspace_byte_ / sizeof(float) + 1;

        cudaMalloc((void**)&forward_workspace_ptr_, forward_workspace_byte_);
        cudaMalloc((void**)&backward_workspace_ptr_, backward_workspace_byte_);


  }

  	void ReadData(std::istream &is){

  		ExpectToken(is, "<LearnRateCoef>") ;
  		ReadBasicType(is, &learn_rate_coef_);
  		ExpectToken(is, "<BiasLearnRateCoef>");
  		ReadBasicType(is, &bias_learn_rate_coef_);
  		ExpectToken(is, "<FmapXLen>") ;
  		ReadBasicType(is, &param_.fmap_input_w);
  		ExpectToken(is, "<FmapYLen>") ;
  		ReadBasicType(is, &param_.fmap_input_h);
  		ExpectToken(is, "<FiltXLen>");
  		ReadBasicType(is, &param_.kernel_w);
  		ExpectToken(is, "<FiltYLen>");
  		ReadBasicType(is, &param_.kernel_h);
  		ExpectToken(is, "<FiltXStep>");
  		ReadBasicType(is, &param_.stride_w);
  		ExpectToken(is, "<FiltYStep>");
  		ReadBasicType(is, &param_.stride_h);
  		ExpectToken(is, "<FmapXPad>");
  		ReadBasicType(is, &param_.pad_w);
  		ExpectToken(is, "<FmapYPad>");
  		ReadBasicType(is, &param_.pad_h);


      assert( (param_.fmap_input_h + 2*param_.pad_h - param_.kernel_h)%param_.stride_h == 0 );
      assert( (param_.fmap_input_w + 2*param_.pad_w - param_.kernel_w)%param_.stride_w == 0 );
      param_.fmap_output_h = (param_.fmap_input_h + 2*param_.pad_h - param_.kernel_h)/param_.stride_h + 1 ;
      param_.fmap_output_w = (param_.fmap_input_w + 2*param_.pad_w - param_.kernel_w)/param_.stride_w + 1 ;
      assert(dim_out_ % (param_.fmap_output_w * param_.fmap_output_h) == 0) ;
      assert(dim_in_ % (param_.fmap_input_w * param_.fmap_input_h) == 0) ;
      param_.num_output_filter = dim_out_ / (param_.fmap_output_w * param_.fmap_output_h) ;
      param_.num_input_filter = dim_in_ / (param_.fmap_input_w * param_.fmap_input_h) ;
      
      // weights
      Filter_.Resize(param_.num_output_filter, param_.num_input_filter * param_.kernel_h * param_.kernel_w);
      Filter_updata_.Resize(param_.num_output_filter, param_.num_input_filter * param_.kernel_h * param_.kernel_w);
      Bias_.Resize(param_.num_output_filter);
      Bias_updata_.Resize(param_.num_output_filter);
  		ExpectToken(is, "<Filters>") ; 
      Filter_.Read(is);
      ExpectToken(is, "<Bias>") ;
      Bias_.Read(is);


  	}

  	void WriteData(std::ostream &os) {
      // weights
  		  WriteToken(os, "<LearnRateCoef>");
  		  WriteBasicType(os, learn_rate_coef_);
  		  WriteToken(os, "<BiasLearnRateCoef>");
  		  WriteBasicType(os, bias_learn_rate_coef_);
  		  WriteToken(os, "<FmapXLen>");
  		  WriteBasicType(os, param_.fmap_input_w);
  		  WriteToken(os, "<FmapYLen>");
  		  WriteBasicType(os, param_.fmap_input_h);
  		  WriteToken(os, "<FiltXLen>");
  		  WriteBasicType(os, param_.kernel_w);
  		  WriteToken(os, "<FiltYLen>");
  		  WriteBasicType(os, param_.kernel_h);
  		  WriteToken(os, "<FiltXStep>");
  		  WriteBasicType(os, param_.stride_w);
  		  WriteToken(os, "<FiltYStep>");
  		  WriteBasicType(os, param_.stride_h);
  		  WriteToken(os, "<FmapXPad>");
  		  WriteBasicType(os, param_.pad_w);
  		  WriteToken(os, "<FmapYPad>");
  		  WriteBasicType(os, param_.pad_h);

  		  WriteToken(os, "<Filters>");
        Filter_.Write(os);
        WriteToken(os, "<Bias>");
        Bias_.Write(os);
  	}

	void ForwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *output){

          if(!Init_){
            Init();
            Init_ = true ;
          }

      		BaseFloat alpha = 1.0f;
      		BaseFloat beta = 0.0f;
      		const BaseFloat *data_ptr = input.Data();
      		const BaseFloat *wmat_ptr = Filter_.Data();
      		BaseFloat *out_ptr = output->Data();
      		const BaseFloat *bias_ptr = Bias_.Data();

      		CHECK_EQ(cudnnConvolutionForward(handle_,
                                       		&alpha,
                                       		in_desc_,
                                       		data_ptr,
                                       		filter_desc_,
                                       		wmat_ptr  ,
                                       		conv_desc_,
                                       		algo_,
                                       		forward_workspace_ptr_,
                                       		forward_workspace_byte_,
                                       		&beta,
                                       		out_desc_,
                                       		out_ptr ), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
          
        	beta = 1.0f;
	        CHECK_EQ(cudnnAddTensor(handle_,
                                  	&alpha,
                                  	bias_desc_,
                                  	bias_ptr ,
                                  	&beta,
                                  	out_desc_,
                                  	out_ptr  ), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);
    
	}

	void BackwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *input_diff, cuMatrix<BaseFloat> &output, cuMatrix<BaseFloat> &output_diff){
    
      		BaseFloat alpha = 1.0f;
      		BaseFloat beta = 0.0f;
          BaseFloat mmt = opts_.momentum ;
      		BaseFloat* grad_ptr = output_diff.Data() ;
      		BaseFloat* gdata_ptr = input_diff->Data() ;
      		BaseFloat* gbias_ptr = Bias_updata_.Data() ;
      		BaseFloat* data_ptr = input.Data() ;
      		BaseFloat* gwmat_ptr = Filter_updata_.Data() ;
      		BaseFloat* wmat_ptr = Filter_.Data();
 			


      		CHECK_EQ(cudnnConvolutionBackwardBias(handle_,
                                            		&alpha,
                                            		out_desc_,
                                            		grad_ptr  ,
                                            		&mmt ,
                                            		bias_desc_,
                                            		gbias_ptr  ),
                 									CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

      		CUDNN_CALL(cudnnConvolutionBackwardFilter(handle_,
               											&alpha,
               											in_desc_,
               											data_ptr  ,
               											out_desc_,
               											grad_ptr  ,
               											conv_desc_,
               											back_algo_w_,
               											backward_workspace_ptr_,
               											backward_workspace_byte_,
               											&mmt ,
               											filter_desc_,
               											gwmat_ptr  ));
      		CHECK_EQ(cudnnConvolutionBackwardData(handle_,
               										&alpha,
               										filter_desc_,
               										wmat_ptr  ,
               										out_desc_,
               										grad_ptr  ,
               										conv_desc_,
               										back_algo_,
               										backward_workspace_ptr_,
               										backward_workspace_byte_,
               										&beta,
               										in_desc_,
               										gdata_ptr  ), CUDNN_STATUS_SUCCESS, __FILE__, __LINE__);

	}	

	void Update(cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &output_diff){

		BaseFloat lr = opts_.learn_rate ;
    BaseFloat l2_penalty = opts_.l2_penalty;
    Filter_updata_.Scale(1.0/(param_.fmap_output_h * param_.fmap_output_w));
    Bias_updata_.Scale(1.0/(param_.fmap_output_h*param_.fmap_output_w));
		Filter_updata_.Scale(-lr * learn_rate_coef_);
		Bias_updata_.Scale(-lr * bias_learn_rate_coef_);

    if (l2_penalty != 0.0) {
        Filter_.AddMat(Filter_, kNoTrans, -lr*l2_penalty*input.NumRows());
    }


    Filter_.AddMat( Filter_updata_, kNoTrans, 1.0);
    Bias_.AddVec(Bias_updata_ , 1.0);

	}


private:
	 /*data*/
	 cuMatrix<BaseFloat> Filter_ ;
	 cuVector<BaseFloat> Bias_ ;

	 cuMatrix<BaseFloat> Filter_updata_;
	 cuVector<BaseFloat> Bias_updata_;
    bool Init_ ;
  	cudnnHandle_t handle_ ;
  	cudaStream_t stream_ ;
  	size_t forward_workspace_;
  	size_t backward_workspace_;
  	size_t forward_workspace_byte_;
  	size_t backward_workspace_byte_;
  	size_t bias_offset_;
  	BaseFloat* forward_workspace_ptr_ ;
    BaseFloat* backward_workspace_ptr_ ;
  	BaseFloat learn_rate_coef_;
  	BaseFloat bias_learn_rate_coef_;
  	cudnnTensorDescriptor_t in_desc_;
  	cudnnTensorDescriptor_t out_desc_;
  	cudnnTensorDescriptor_t bias_desc_;
  	cudnnFilterDescriptor_t filter_desc_;
  	cudnnConvolutionDescriptor_t conv_desc_;
  	cudnnConvolutionFwdAlgo_t algo_;
  	cudnnConvolutionBwdDataAlgo_t back_algo_;
  	cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  	cudnnTensorFormat_t format_;
  	ConvolutionParam param_;

};

}//end of namespace

#endif//NNET_CUDNN_CONVNETCOMPONENTCUDNN_H_
