//Matrix/Matrix.h
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
#ifndef MATRIX_MATRIX_H_
#define MATRIX_MATRIX_H_

#include "../base/common.h"
#include "../base/nnet-math.h"
#include "Vector.h"
#include "Kernel.h"

namespace nnet{

template<typename Real> class Vector;

template<typename Real>
class Matrix{
public:

	Matrix(){}

	Matrix(int32 Rows, int32 Cols);

	~Matrix();

	Matrix& operator=( Matrix& rhs);

	void CopyFromPtr(Real* data, int32 length);

	void Vector2Matrix(Vector<Real>& V) ;

	bool SamDim( Matrix<Real> &A, MatrixTransposeType TransA);

	Real Sum();

	void Set(Real value);

	void Resize(int32 Row, int32 Col);

	void Transform();

	inline Real* Data() {return data_;}

	inline int32 NumRows() {return Rows_;}

	inline int32 NumCols() {return Cols_;}
	// *this = alpha*po(A)*po(B) + beta*(*this)
	void AddMatMat( Matrix<Real> &A, MatrixTransposeType TransA,  Matrix<Real> &B, MatrixTransposeType TransB, Real alpha, Real beta);

	void Sigmoid(Matrix<Real> &A, MatrixTransposeType TransA);

	void Tanh(Matrix<Real> &A, MatrixTransposeType TransA);

	void ReLU(Matrix<Real> &B, MatrixTransposeType TransA);

	void DiffSigmoid( Matrix<Real> &out,  Matrix<Real> &out_diff);

	void DiffTanh( Matrix<Real> &out,  Matrix<Real> &out_diff);

	void DiffReLU( Matrix<Real> &out,  Matrix<Real> &out_diff);

	void SubMat( Matrix<Real> &A, MatrixTransposeType TransA);

	void AddMat( Matrix<Real> &A, MatrixTransposeType TransA, Real alpha);

	void BinarizeProbs(Real value);

	void Normalized_Cmvn();

	void Random(Real param_stddev);

	void Read(std::istream &is) ;

	void Write(std::ostream &os) ;

	inline const Real operator() (int32 r , int32 c) const{

		assert(r < Rows_ && c < Cols_) ;
		return *(data_ + r * Cols_ + c) ;
	}

	inline Real& operator() (int32 r, int32 c){

		assert(r < Rows_ && c < Cols_) ;
		return *(data_ + r * Cols_ + c) ;
	}



private:

	Real* data_;

	int32 Rows_;

	int32 Cols_;

	//Disallow_Copy_And_Assign(Matrix);
 };
}
#endif