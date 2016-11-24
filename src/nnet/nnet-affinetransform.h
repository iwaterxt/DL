//nnet/nnet-affinetransform.h
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
#ifndef NNET_NNET_AFFINETRANSFORM_H_
#define NNET_NNET_AFFINETRANSFORM_H_

#include <math.h>
#include "nnet-component.h"
#include "../base/nnet-math.h"

namespace nnet{


class AffineTransform : public UpdateComponent{
public:
	AffineTransform(int32 dim_in, int32 dim_out):UpdateComponent(dim_in, dim_out),
      linearity_(dim_out, dim_in), bias_(dim_out),
	linearity_update_(dim_out, dim_in), bias_update_(dim_out),delta_(dim_out, dim_in),mmt_(0.0)
	{
      }
	~AffineTransform(){};

  ComponentType GetType() const{
        return kAffineTransform;
  }

  Component* Copy() const { return new AffineTransform(*this); }

	void InitData(std::istream &is) {
   		// define options
    	float bias_mean = 0.0, bias_range = 0.0, param_stddev = 0.0;
    	// parse config
    	std::string token; 
    	while (!is.eof()) {
      		ReadToken(is, token); 
      		/**/ if (token == "<ParamStddev>") ReadBasicType(is, &param_stddev);
      		else if (token == "<BiasMean>")    ReadBasicType(is, &bias_mean);
      		else if (token == "<BiasRange>")   ReadBasicType(is, &bias_range);
      		else std::cout << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange)";
      		is >> std::ws; // eat-up whitespace
    	}

    	Matrix<BaseFloat> mat(dim_out_, dim_in_);
    	for (int32 r=0; r<dim_out_; r++) {
     	 	for (int32 c=0; c<dim_in_; c++) {
       	 	mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
                   
      		}
    	}

    	linearity_ = mat;
    	//
    	Vector<BaseFloat> vec(dim_out_);
    	for (int32 i=0; i<dim_out_; i++) {
      	// +/- 1/2*bias_range from bias_mean:
      	 vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    	}

    	   bias_ = vec;

    	}

  	void ReadData(std::istream &is){
      // weights
        linearity_.Read(is);
        bias_.Read(is);

        assert(linearity_.NumRows() == this->OutputDim());
        assert(linearity_.NumCols() == this->InputDim());
        assert(bias_.Dim() == this->OutputDim());
  	}

  	void WriteData(std::ostream &os) {
      // weights
      linearity_.Write(os);
      bias_.Write(os);
  	}


	void ForwardPropegation( cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *output){

        output->AddVecToRows(bias_);
            
	      output->AddMatMat(input , kNoTrans , linearity_ , kTrans , 1.0 , 0.0);


	}

	void BackwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat> *input_diff, cuMatrix<BaseFloat> &output, cuMatrix<BaseFloat> &output_diff){
            input_diff->AddMatMat(output_diff, kNoTrans, linearity_, kNoTrans, 1.0, 0);
	}

//SGD
  /*
	void Update( cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &output_diff){

            //BaseFloat momentum = opts_.momentum ;

            BaseFloat learn_rate = opts_.learn_rate ;

            BaseFloat l2_penalty = opts_.l2_penalty ;

            BaseFloat bias_learn_rate = opts_.bias_learn_rate ;

		        assert(linearity_.NumRows() == linearity_update_.NumRows()&&linearity_.NumCols() == linearity_update_.NumCols());
		
            assert(bias_.Dim() == bias_update_.Dim());

		        BaseFloat learn_scale = 1.0 / input.NumRows();

            linearity_update_.AddMatMat(output_diff, kTrans, input, kNoTrans,  -learn_scale * learn_rate, mmt_);

		        bias_update_.SumColMat(output_diff);

            
            bias_update_.Scale(- bias_learn_rate / input.NumRows());

                // l2 regularization
            if (l2_penalty != 0.0) {
              linearity_.AddMat(linearity_, kNoTrans, -learn_rate*l2_penalty*input.NumRows());
            }

		        //update
		        linearity_.AddMat( linearity_update_, kNoTrans, 1.0);

		        bias_.AddVec(bias_update_, 1.0);
            mmt_ = opts_.momentum ;


	}
  */
//Nesterov SGD
  void Update( cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &output_diff){

            //BaseFloat momentum = opts_.momentum ;

            BaseFloat learn_rate = opts_.learn_rate ;

            BaseFloat l2_penalty = opts_.l2_penalty ;

            BaseFloat bias_learn_rate = opts_.bias_learn_rate ;

            assert(linearity_.NumRows() == linearity_update_.NumRows()&&linearity_.NumCols() == linearity_update_.NumCols());
    
            assert(bias_.Dim() == bias_update_.Dim());



            linearity_update_.AddMatMat(output_diff, kTrans, input, kNoTrans,  -learn_rate, 0.0);


            linearity_.AddMat(linearity_update_, kNoTrans, 1+mmt_);
            linearity_.AddMat(delta_, kNoTrans, mmt_*mmt_);
            delta_.Scale(mmt_);
            delta_.AddMat(linearity_update_, -learn_rate);
            bias_update_.SumColMat(output_diff);
                // l2 regularization
            if (l2_penalty != 0.0) {
              linearity_.AddMat(linearity_, kNoTrans, -learn_rate*l2_penalty*input.NumRows());
            }
            bias_.AddVec(bias_update_, 1.0);
            mmt_ = opts_.momentum ;
  }


private:
	/*data*/
	cuMatrix<BaseFloat> linearity_ ;
	cuVector<BaseFloat> bias_ ; 

	cuMatrix<BaseFloat> linearity_update_ ;
	cuVector<BaseFloat> bias_update_ ;
  cuMatrix<BaseFloat> delta_ ;
  BaseFloat mmt_ ;
};
}//namespace nnet

#endif