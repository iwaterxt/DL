//nnet-convnetcomponent.h
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

#include "nnet-nnet.h"

namespace nnet{
	
	Nnet::~Nnet() {
		Destroy() ;
	}
	void Nnet::Check() const{
		for(int32 i = 0; i < Components_.size()-2; i++)
			assert(Components_[i+1]->InputDim() == Components_[i]->OutputDim());

	}

	void Nnet::AppendComponent(Component* component) {
		    //initial component
        Components_.push_back(component);

        propagate_buf_.resize(NumComponent() + 1) ;

        backpropagate_buf_.resize(NumComponent() + 1) ;

       // Check();
    }

	//Forward of the network
	void Nnet::ForwardPropegation(cuMatrix<BaseFloat> &input, cuMatrix<BaseFloat>* output){
		propagate_buf_[0] = input;
		int32 i = 0;
		for(; i < Components_.size(); i++)
			Components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
		//Components_[i]->Propagate(propagate_buf_[i], output);
            *output = propagate_buf_[Components_.size()] ;
           // propagate_buf_[i+1] = *output ;
	}

	//Backword of the network
	void Nnet::BackwardPropegation(cuMatrix<BaseFloat> &output_diff, cuMatrix<BaseFloat>* input_diff){
		backpropagate_buf_[Components_.size()] = output_diff;

             for(int32 j = Components_.size()-1 ; j >= 0; j--){

                Components_[j]->Backpropagate(propagate_buf_[j], &backpropagate_buf_[j], propagate_buf_[j+1], backpropagate_buf_[j+1]);
                if(Components_[j]->IsUpdatable()){
                    UpdateComponent *uc = dynamic_cast<UpdateComponent*>(Components_[j]);
                    uc->Update(propagate_buf_[j], backpropagate_buf_[j+1]) ;
                }
            }
            if (NULL != input_diff) (*input_diff) = backpropagate_buf_[0];
	}

	void Nnet::Feedforward( cuMatrix<BaseFloat> &in, cuMatrix<BaseFloat> *out) {
  		assert(NULL != out);

  		if (NumComponent() == 0) { 
    		    out->Resize(in.NumRows(), in.NumCols());
    		    out->CopyFromMat(in); 
    		    return; 
  		}

  		if (NumComponent() == 1) {
   			Components_[0]->ForwardPropegation(in, out);
    		      return;
  		}

  	// we need at least 2 input buffers
  		assert(propagate_buf_.size() >= 2);
  	// ForwardPropegation by using exactly 2 auxiliary buffers
  		int32 L = 0;
  		Components_[L]->ForwardPropegation(in, &propagate_buf_[L%2]);

  		for(; L<=NumComponent()-2; L++) {
    		Components_[L]->ForwardPropegation(propagate_buf_[(L-1)%2], &propagate_buf_[L%2]);
  		}
  		Components_[L]->ForwardPropegation(propagate_buf_[(L-1)%2], out);
  	// release the buffers we don't need anymore
  		propagate_buf_[0].Resize(0,0);
  		propagate_buf_[1].Resize(0,0);
  }

	//return the input dimention
	int32 Nnet::InputDim()const{
		return Components_[0]->InputDim();
	}
	//return the output dimention
	int32 Nnet::OutputDim()const{
		return Components_[Components_.size()-1]->OutputDim();
	}
	//return the number of component of Net
	int32 Nnet::NumComponent()const{
		return Components_.size();
	}
	//read net model from nnet model
	void Nnet::Read(std::string &nnet_file) {
		std::ifstream is(nnet_file);
  	// get the network layers from a factory
  		Component *comp = NULL;
  		while (NULL != (comp = comp->Read(is))) {
       
    		if (NumComponent() > 0 && Components_.back()->OutputDim() != comp->InputDim()) {
      			std::cerr << "Dimensionality mismatch!"
                		  << " Previous layer output:" << Components_.back()->OutputDim()
                		  << " Current layer input:" << comp->InputDim();
    		}
    	  Components_.push_back(comp);
  	  }

  	     // create empty buffers

  	     propagate_buf_.resize(NumComponent()+1);
  	     backpropagate_buf_.resize(NumComponent()+1);
           
  	     Check(); //check consistency (dims...)
  }
	 //write net model into a file
	void Nnet::Write(std::string &file){
		std::ofstream os(file) ;
		WriteToken(os, "<NnetProto>") ;
		WriteToken(os, "\n") ;
		for(int32 i = 0; i < Components_.size() ; i++){
			Components_[i]->Write(os) ;
		}
		WriteToken(os, "</NnetProto>") ;
	}
	//initial with configure file
	void Nnet::Init(const std::string &file){
		std::ifstream is(file);
            if (!is)
          {
                std::cerr << "Error : cound not to open file : " << file << std::endl ;
                return ;
          }
		is >> std::ws; //cousume whitespace
		ExpectToken(is, "<NnetProto>");
		std::string conf_line;
		while(1){
			if(is.eof()){
				std::cout<<"missing </NnetProto> at the end of file!"<<std::endl;
				break;
			}
			is >> std::ws;//cousume whitespace
			assert(is.good());
			if(CheckToken(is,'/')){
				ExpectToken(is,"</NnetProto>");
				break;
			}
			std::getline(is,conf_line);
			Component* ans = NULL ;
			ans = ans->Init(conf_line) ;
      printf("the input_dim of component is : %d\n", ans->InputDim());
			AppendComponent(ans) ;

		}
            
		is.close();
		Check();
	}

  void Nnet::SetTrnOption(NnetTrainOptions &opt){

      TrnOptions_.learn_rate = opt.learn_rate ;
      TrnOptions_.bias_learn_rate = opt.bias_learn_rate ;
      TrnOptions_.momentum = opt.momentum ;
      TrnOptions_.l2_penalty = opt.l2_penalty ;
      TrnOptions_.l1_penalty = opt.l1_penalty ;
      TrnOptions_.apply_norm = opt.apply_norm ;
      TrnOptions_.image_size = opt.image_size ;
      TrnOptions_.tr_number = opt.tr_number ;
      TrnOptions_.cv_number = opt.cv_number ;
      TrnOptions_.test_number = opt.test_number ;
      TrnOptions_.class_number = opt.class_number ;
      TrnOptions_.minibatch_size = opt.minibatch_size ;
      TrnOptions_.is_train = opt.is_train ;

      for(int32 j = Components_.size()-1 ; j >= 0; j--){
        /*
           if (Components_[j]->IsUpdatable()){
                    UpdateComponent *uc = dynamic_cast<UpdateComponent*>(Components_[j]) ;
                    uc->SetTrnOptions(TrnOptions_) ;
           }
        */
           Component *uc = dynamic_cast<Component*> (Components_[j]);
           uc->SetTrnOptions(TrnOptions_);
      }

  }

   void Nnet::Destroy(){
  	  Components_.resize(0);
  	  propagate_buf_.resize(0);
  	  backpropagate_buf_.resize(0);
  }

	void Nnet::Train(float* train_set, float* train_label, int32 num_train_samples, std::string objective_function){
  		Xent xent;
  		Mse mse;
  		int size=num_train_samples;
            int data_length = TrnOptions_.image_size ;
  		int batch_size = TrnOptions_.minibatch_size;
  		int batch_start, batch_end;
  		unsigned int num_batch = (0 == (size % batch_size))? (size / batch_size):(size / batch_size) + 1;
  		Matrix<float> data_batch(batch_size , data_length) ;
  		Vector<float> labels(batch_size);
  		Matrix<float> data_label(batch_size , TrnOptions_.class_number);
            cuMatrix<float> nnet_out(batch_size, TrnOptions_.class_number);
            cuMatrix<float> obj_diff(batch_size, TrnOptions_.class_number);
            cuMatrix<float> nnet_in(batch_size, data_length);
            cuMatrix<float> nnet_tgt(batch_size,TrnOptions_.class_number);
        
      	for(unsigned int batch = 0;batch < num_batch;++batch)
      	{
        		batch_start = batch * batch_size ;
        		batch_end = std::min(batch_start + batch_size, size);
        		data_batch.CopyFromPtr((train_set + batch_start * Components_[0]->InputDim()) , static_cast<int>(batch_end - batch_start)*data_length);
                   nnet_in = data_batch;//this sentence should be changed
        		labels.CopyFromPtr((train_label + batch_start), (batch_end - batch_start));
        		data_label.Vector2Matrix(labels);
        		nnet_tgt = data_label;
        		data_label.Set(0.0);

                  //option
                  if(TrnOptions_.apply_norm)
                        nnet_in.ApplyNorm();
        		//network forward pass
        		ForwardPropegation(nnet_in, &nnet_out);
        		//caculate the object_function
         		if (objective_function == "Xent") {
          		  xent.Eval(nnet_out, nnet_tgt, &obj_diff);
        		} else if (objective_function == "mse") {
          		  mse.Eval(nnet_out, nnet_tgt, &obj_diff);
                     //printf("the sum of nnet_tgt  is: %f, %s\n", nnet_tgt.Sum(), __FILE__);
                     //printf("the sum of  nnet_out is: %f, %s\n", nnet_out.Sum(), __FILE__);
        		} else {
          		  std::cout << "Unknown objective function code : " << objective_function;
        		}

        		BackwardPropegation(obj_diff, NULL);

      	}
    	     
	}
  void Nnet::CrossValidate(float* dev_set, float* dev_label, int32 num_dev_samples, std::string objective_function){
          Xent xent;
          Mse mse;
          int size=num_dev_samples;
          int data_length = TrnOptions_.image_size ;
          int batch_size = TrnOptions_.minibatch_size;
          int batch_start, batch_end;
          unsigned int num_batch = (0 == (size % batch_size))? (size / batch_size):(size / batch_size) + 1;
          Matrix<float> data_batch(batch_size , Components_[0]->InputDim()) ;
          Vector<float> labels(batch_size);
                        
          Matrix<float> data_label(batch_size , TrnOptions_.class_number);
          cuMatrix<float> nnet_out(batch_size, TrnOptions_.class_number);
          cuMatrix<float> obj_diff(batch_size, TrnOptions_.class_number);
          cuMatrix<float> nnet_in(batch_size, data_length);
          cuMatrix<float> nnet_tgt(batch_size,TrnOptions_.class_number);
          for(unsigned int batch = 0;batch < num_batch;++batch){
                batch_start = batch * batch_size ;
                batch_end = std::min(batch_start + batch_size, size);
                data_batch.CopyFromPtr((dev_set + batch_start * Components_[0]->InputDim()) , static_cast<int>(batch_end - batch_start)*data_length);
                nnet_in = data_batch;//this sentence should be changed
                labels.CopyFromPtr((dev_label + batch_start), (batch_end - batch_start));
                data_label.Vector2Matrix(labels);
                nnet_tgt = data_label;
                data_label.Set(0.0);

                //option
                if(TrnOptions_.apply_norm)
                      nnet_in.ApplyNorm();
                      //network forward pass
                ForwardPropegation(nnet_in, &nnet_out);
                      //caculate the object_function
                if (objective_function == "Xent") {
                    xent.Eval(nnet_out, nnet_tgt, &obj_diff);
                } else if (objective_function == "mse") {
                    mse.Eval(nnet_out, nnet_tgt, &obj_diff);
                } else {
                    std::cout << "Unknown objective function code : " << objective_function;
                }
          }
  }

	void Nnet::Predict(float* test_set, float* test_label, int32 num_test_samples){
  		
      int size = num_test_samples;
      int data_length = TrnOptions_.image_size;
  		int batch_size = TrnOptions_.minibatch_size;
  		int error_count = 0;
  		float error_rate = 0.0;
  		int batch_start, batch_end;
  		unsigned int num_batch = (0 == (size % batch_size))? (size / batch_size):(size / batch_size) + 1;

  		//std::string final_nnet = "../model/final.nnet" ;
  		//nnet.Read(final_nnet);
  		cuMatrix<float> nnet_out, nnet_in;
  		nnet_in.Resize(batch_size, data_length);
  		Matrix<float> data_batch(batch_size, data_length);
  		Vector<float> labels(batch_size);
  		Vector<float> v(batch_size);
  		for(unsigned int batch = 0;batch < num_batch;++batch)
    	     {
      		  batch_start = batch * batch_size;
      		  batch_end = std::min(batch_start + batch_size, size);
      		  data_batch.CopyFromPtr(const_cast<float*>(test_set + batch_start * data_length) , static_cast<int>(batch_end - batch_start)*data_length);
      		  nnet_in = data_batch;
                    if(TrnOptions_.apply_norm)
                      nnet_in.ApplyNorm();
      		  labels.CopyFromPtr(const_cast<float*>(test_label + batch_start), (batch_end - batch_start));
      		  ForwardPropegation(nnet_in, &nnet_out);
                    cuVector<float> max_id(nnet_out.NumRows()) ;
      		  nnet_out.FindRowMaxId(max_id);
      		  max_id.CopyToVector(v);
      		  max_id.Resize(batch_size);
      		  for(int i = 0; i < batch_size; i++)
      		  {
        		    if((int)v(i) != (int)labels(i))
          		            error_count++;
      		  }   
    	     }
  		error_rate = (float)error_count/(float)size;
  		printf("the error_rate is :%f\n",error_rate);

	}

}//end of namespace