//nnet/nnet-component.h
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
# ifndef NNET_NNET_COMPONENT_H_
# define NNET_NNET_COMPONENT_H_
# include <algorithm>
# include "../base/common.h"
# include "../cumatrix/cuMatrix.h"
# include "../base/nnet-io.h"
# include "nnet-train-options.h"
namespace nnet{

class Component{
public:
	typedef enum{
		kUnknown = 0x0,

		kUpdatableComponent = 0x0100,
		kAffineTransform,
    kConvnetComponentCudnn,
    kCudnnMaxPooling2DComponent,
    kBatchNormCudnn,
		kActiveFunction = 0x0200,
		kSigmoid,
		kTanh,
		kReLU,
		kDropOut,
		kSoftMax,
    kCudnnSoftMax
		
	} ComponentType;

	Component(int32 input_dim, int32 output_dim):dim_in_(input_dim), dim_out_(output_dim) {}
	~Component(){}
	struct key_value {
   	 const Component::ComponentType key;
   	 const char* value;
 	};
  /// Mapping of types and markers (the table is defined in nnet-component.cc) 
 	static const struct key_value kMarkerMap[];
  /// Convert component type to marker
  	static const char* TypeToMarker(ComponentType t);
  /// Convert marker to component type (case insensitive)
  	static ComponentType MarkerToType(const std::string &s);

  	Component* NewComponentOfType(ComponentType comp_type, int32 input_dim, int32 output_dim);

	Component* Init(std::string &is);

	int32 InputDim(){

		return dim_in_;
	}

	int32 OutputDim(){
		return dim_out_;
	}

	virtual ComponentType GetType() const= 0;

	virtual Component* Copy() const = 0;

	Component*  Read(std::ifstream &is);

  void SetTrnOptions(NnetTrainOptions &opt){

      opts_.learn_rate = opt.learn_rate ;
      opts_.bias_learn_rate = opt.bias_learn_rate ;
      opts_.momentum = opt.momentum ;
      opts_.l2_penalty = opt.l2_penalty ;
      opts_.l1_penalty = opt.l1_penalty ;
      opts_.minibatch_size = opt.minibatch_size ;
      opts_.is_train = opt.is_train ;
  }

	void Write(std::ofstream &os);

	/// Perform forward pass propagation Input->Output
  void Propagate( cuMatrix<BaseFloat> &in, cuMatrix<BaseFloat> *out); 
  /// Perform backward pass propagation, out_diff -> in_diff
  /// '&in' and '&out' will sometimes be unused... 
  void Backpropagate( cuMatrix<BaseFloat> &in,
                      cuMatrix<BaseFloat> *in_diff,
                      cuMatrix<BaseFloat> &out,
                      cuMatrix<BaseFloat> &out_diff); 

  virtual void ForwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*output)= 0;

	virtual void BackwardPropegation(cuMatrix<BaseFloat>&input, cuMatrix<BaseFloat>*input_diff, cuMatrix<BaseFloat>&output, cuMatrix<BaseFloat> &output_diff) = 0;

	virtual bool IsUpdatable() = 0 ;
	/// Initialize internal data of a component
  virtual void InitData(std::istream &is) { }

  /// Reads the component content
  virtual void ReadData(std::istream &is) { }

  /// Writes the component content
  virtual void WriteData(std::ostream &os) { }
  
public:

  NnetTrainOptions opts_; 

protected:
	/* data */
	int32 dim_in_;
	int32 dim_out_;
};


class UpdateComponent : public Component{
public:
	UpdateComponent(int32 dim_in, int32 dim_out):Component(dim_in, dim_out)
	{}

	virtual ~UpdateComponent(){}

	bool IsUpdatable(){
	 	
	 	return true;
	}

	ComponentType GetType(){

		return kUpdatableComponent ;
	}



	// Compute gradient and update parameters
  	virtual void Update( cuMatrix<BaseFloat> &input,  cuMatrix<BaseFloat> &diff) = 0;

  	virtual void InitData(std::istream &is) = 0;

};



inline void Component::Propagate( cuMatrix<BaseFloat> &in, cuMatrix<BaseFloat> *out) {
  // Check the dims
  if (dim_in_ != in.NumCols()) {
    std::cout << "Non-matching dims! " << TypeToMarker(GetType()) 
              << " input-dim : " << dim_in_ << " data : " << in.NumCols();
  }
  // Allocate target buffer
    out->Resize(in.NumRows(), dim_out_); // reset
  // Call the propagation implementation of the component
  ForwardPropegation(in, out);
}


inline void Component::Backpropagate( cuMatrix<BaseFloat> &in,
			                                cuMatrix<BaseFloat> *in_diff,
                                      cuMatrix<BaseFloat> &out,
                                      cuMatrix<BaseFloat> &out_diff
                                     ) {
  // Check the dims
  if (dim_out_ != out_diff.NumCols()) {
    
    std::cout << "Non-matching output dims, component:" << dim_out_ 
              << " data:" << out_diff.NumCols();
  }
  
  // Target buffer NULL : backpropagate only through components with nested nnets.
  if (in_diff == NULL) {

      return;
  } else {
  	//printf("the value of dim_in_ is : %d, the value of dim_out_ is : %d\n", dim_in_, dim_out_);
    // Allocate target buffer
    in_diff->Resize(out_diff.NumRows(), dim_in_); // reset
    // Asserts on the dim
    assert((in.NumRows() == out.NumRows()) &&
                 (in.NumRows() == out_diff.NumRows()) &&
                 (in.NumRows() == in_diff->NumRows()));
    assert(in.NumCols() == in_diff->NumCols());
    assert(out.NumCols() == out_diff.NumCols());
    // Call the backprop implementation of the component
    BackwardPropegation(in, in_diff, out , out_diff);
  }
}

}//namespace nnet

#endif
