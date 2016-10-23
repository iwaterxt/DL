//cuMatrix/cuMatrix.h
// Copyright 2014-12-31   (Author: xutao)
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
#ifndef CUMATRIX_CUMATRIX_H_
#define CUMATRIX_CUMATRIX_H_

#include "../base/common.h"
#include "../matrix/Matrix.h"
#include "cuVector.h"
#include "../matrix/Vector.h"
#include "../base/nnet-io.h"
#include <stdlib.h>

namespace nnet{

template<typename Real> class cuVector;
template<typename Real> class Matrix;
template<typename Real> class Vector;

template<typename Real> 
class cuMatrix{
public:

	cuMatrix();

	cuMatrix(int32 Rows, int32 Cols);

	~cuMatrix();

	inline int32 NumRows(){return Rows_;}

	inline int32 NumCols(){return Cols_;}

	inline size_t NumStride(){return Stride_;}

	inline Real* Data() {return data_;}

	void FindRowMaxId(cuVector<Real> &Vec);

	void CopyFromMat(cuMatrix<Real> &Mat);

	void CopyFromMat(Matrix<Real> &Mat);

	void CopyToMat(Matrix<Real> &Mat);

	void CopyToHostPtr(Real* Ptr);

	void SumRowMatplusexp(cuVector<Real> &Vec);

	void ApplySoftMaxPerRow();

	void Sigmoid(cuMatrix<Real> &input);

	void DiffSigmoid(cuMatrix<Real> &out, cuMatrix<Real> &out_diff);

	void Tanh(cuMatrix<Real> &input);

	void DiffTanh(cuMatrix<Real> &out, cuMatrix<Real> &out_diff);

	void ReLU(cuMatrix<Real> &input);

	void DiffReLU(cuMatrix<Real> &out, cuMatrix<Real> &out_diff);

	void ApplyFloor(Real value);

	void ApplyNorm();

	void ApplyHeaviside();

	void AddVecToRows(cuVector<Real> &Vec) ;

	void MulElements(cuMatrix<Real> &Mat) ;

	void Resize(int32 Rows, int32 Cols) ;

	void Set(Real value);

	void Log();

	void Exp();

	Real Sum();
	//L1
	Real AbSum();

	void Scale(Real scale);

	void BinarizeProbs(Real Probs);

	void AddMatMat( cuMatrix<Real> &A, MatrixTransposeType TransA,  cuMatrix<Real> &B, MatrixTransposeType TransB, Real alpha, Real beta);

	void AddMat( cuMatrix<Real> &Mat, MatrixTransposeType TransM, Real alpha);

	void Read(std::istream &is) ;

	void Write(std::ostream &os) ;

	void check() ;

	void operator=(Matrix<Real> &Mat) ;

	void operator=(cuMatrix<Real> &Mat) ;

private:

	Real* data_;

	int32 Rows_;

	int32 Cols_;

	size_t Stride_ ;

	//Disallow_Copy_And_Assign(cuMatrix);
 };
	
} //namespace nnet
#endif
