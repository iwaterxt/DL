//base/nnet-math.h
// Copyright 2014-11-24   (Author: xutao)
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
#ifndef BASE_NNET_MATH_H_
#define BASE_NNET_MATH_H_

#include <cmath>
#include <limits>
#include <vector>
#include "../base/common.h"

#ifndef DBL_EPSILON
#  define DBL_EPSILON 2.2204460492503131e-16
#endif
#ifndef FLT_EPSILON
#  define FLT_EPSILON 1.19209290e-7f
#endif

#ifndef M_PI
#  define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_SQRT2
#  define M_SQRT2 1.4142135623730950488016887
#endif


#ifndef M_2PI
#  define M_2PI 6.283185307179586476925286766559005
#endif

#ifndef M_SQRT1_2
#  define M_SQRT1_2 0.7071067811865475244008443621048490
#endif

#ifndef M_LOG_2PI
#  define M_LOG_2PI 1.8378770664093454835606594728112
#endif

#ifndef M_LN2
#  define M_LN2 0.693147180559945309417232121458
#endif

#  define NET_ISNAN std::isnan
#  define NET_ISINF std::isinf
#  define NET_ISFINITE(x) std::isfinite(x)

namespace nnet{

inline float RandUniform() {  // random intended to be strictly between 0 and 1.
  return static_cast<float>((rand() + 1.0) / (RAND_MAX+2.0));  
}

inline float RandGauss() {
  return static_cast<float>(sqrt (-2 * std::log(RandUniform())) * cos(2*M_PI*RandUniform()));
}

inline bool IsNaN(BaseFloat dat){
 int & ref=*(int *)&dat;
 return (ref&0x7F800000) == 0x7F800000 && (ref&0x7FFFFF)!=0;
}

}

#endif
