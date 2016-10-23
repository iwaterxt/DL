//Matrix/Vector.h
// Copyright 2015-1-30   (Author: xutao)
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
#ifndef MATRIX_VECTOR_H_
#define MATRIX_VECTOR_H_

#include "../base/common.h"
#include "Matrix.h"
#include <string.h>

namespace nnet{
template<typename Real> class Matrix;

template<typename Real>
class Vector{
public:

	Vector(){}

	Vector(int32 dim);
	
	~Vector();

	inline int32 Dim() {return dim_;}

	inline Real* Data() {return data_;}

	Real Sum();

	Real Max();

	int32 MaxIndex();

	void Set(Real value);

	void Resize(int32 dim) ;

	void CopyFromPtr(Real* data, int32 length) ;

	void SubVec(Vector<Real> &A);

	void AddVec(Vector<Real> &A);

	void SumRowMat( Matrix<Real> &Mat, MatrixTransposeType TransA, Real alpha);

	void CopyFromvector(Vector<Real> &Vec);

	//copy row from a cuMatrix of device
	void CopyRowFromMat(Matrix<Real> &Mat, int32 row);
	
	//copy col from a cumatrix of device
	void CopyColFromMat(Matrix<Real> &Mat, int32 col);

	void Read(std::istream &is) ;

	void Write(std::ostream &os) ;

	inline const Real operator() (int32 s ) const {

		assert(s < dim_) ;
		
		return *(data_ + s) ;

	}

	inline Real& operator() (int32 s){

		assert(s < dim_) ;
		return *(data_ + s) ;
	}

private:

	Real* data_;
	int32 dim_;
	//Disallow_Copy_And_Assign(Vector);
	};

}

#endif

