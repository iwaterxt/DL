//nnet/nnet-component.h
// Copyright 2015-1-28   (Author: xutao)
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
# include <iostream>
# include "nnet-component.h"
# include "nnet-active.h"
# include "nnet-affinetransform.h"
# include "nnet-cudnn-convnetcomponent.h"
# include "nnet-cudnn-2d-pooling.h"
# include "nnet-cudnn-active.h"




namespace nnet{

const struct Component::key_value Component::kMarkerMap[] = {
  { Component::kAffineTransform,"<AffineTransform>" },
  { Component::kConvnetComponentCudnn, "<ConvnetComponentCudnn>"},
  { Component::kCudnnMaxPooling2DComponent, "<CudnnMaxPooling2DComponent>"},
  { Component::kSoftMax,"<Softmax>" },
  { Component::kCudnnSoftMax, "<CudnnSoftMax>"},
  { Component::kSigmoid,"<Sigmoid>" },
  { Component::kReLU,"<ReLU>" },
  { Component::kTanh,"<Tanh>" },
  { Component::kActiveFunction, "<ActiveFunction>" },
  { Component::kDropOut,"<Dropout>" },

};


const char* Component::TypeToMarker(ComponentType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
  }
  std::cout << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string &s) {
  std::string s_lowercase(s);
  std::transform(s.begin(), s.end(), s_lowercase.begin(), ::tolower); // lc
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    std::string m(kMarkerMap[i].value);
    std::string m_lowercase(m);
    std::transform(m.begin(), m.end(), m_lowercase.begin(), ::tolower);
    if (s_lowercase == m_lowercase) return kMarkerMap[i].key;
  }
  std::cout << "Unknown marker : '" << s << "'";
  return kUnknown;
}


Component* Component::NewComponentOfType(ComponentType comp_type,
                      int32 input_dim, int32 output_dim) {
  Component *ans = NULL;
  switch (comp_type) {

    case Component::kSoftMax :
      ans = new SoftMax(input_dim, output_dim);
      break;
    case Component::kCudnnSoftMax :
      ans = new CudnnSoftMax(input_dim, output_dim);
      break;
    case Component::kSigmoid :
      ans = new Sigmoid(input_dim, output_dim);
      break;
    case Component::kTanh :
      ans = new Tanh(input_dim, output_dim);
      break;
    case Component::kDropOut :
      ans = new DropOut(input_dim, output_dim); 
      break;
    case Component::kReLU :
      ans = new ReLU(input_dim, output_dim); 
      break;
    case Component::kActiveFunction:
      ans = new ActiveFunction(input_dim, output_dim);
      break;
    case Component::kAffineTransform :
      ans = new AffineTransform(input_dim, output_dim); 
      break;
    case Component::kConvnetComponentCudnn :
      ans = new ConvnetComponentCudnn(input_dim, output_dim);
      break;
    case Component::kCudnnMaxPooling2DComponent :
      ans = new CudnnMaxPooling2DComponent(input_dim, output_dim);
      break;
    case Component::kUnknown :
    default :
      std::cout << "Missing type: " << TypeToMarker(comp_type);
  }
  return ans;
}

 Component* Component::Init(std::string &conf_line){
  std::istringstream is(conf_line);
  is >> std::ws; //consume whitespace.
  std::string component_type_string;
  int32 dim_out , dim_in ;
  // initialize component w/o internal data
  ReadToken(is, component_type_string);
  ComponentType component_type = MarkerToType(component_type_string);
  ExpectToken(is, "<InputDim>");
  ReadBasicType(is, &dim_in); 
  
  ExpectToken(is, "<OutputDim>");
  
  ReadBasicType(is, &dim_out);

  Component *ans = NewComponentOfType(component_type, dim_in, dim_out);

  // initialize internal data with the remaining part of config line
  ans->InitData(is);
  return ans;

}

Component* Component::Read(std::ifstream &is) {
  std::string token;

  int first_char = is.peek();
  if (first_char == EOF) return NULL;
  ReadToken(is, token);
  // Skip optional initial token
  if(token == "<NnetProto>") {
    ReadToken(is, token); // Next token is a Component
  }
  // Finish reading when optional terminal token appears
  if(token == "</NnetProto>") {
    return NULL;
  }
  int32 input_dim = 0 , output_dim = 0 ; 
  ReadBasicType(is, &input_dim); 
  ReadBasicType(is, &output_dim);
  Component *ans = NewComponentOfType(MarkerToType(token), input_dim, output_dim);

  ans->ReadData(is);

  return ans;
}
 
 void Component::Write(std::ofstream &os){

    WriteToken(os, TypeToMarker(GetType()));
   
    WriteToken(os, dim_in_) ;
   
    WriteToken(os, dim_out_) ;

    os << "\n";
   
    this->WriteData(os) ;
    
 }


}//end of space