//Matrix/Vector.cc
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
#include <assert.h>
#include "Vector.h"

namespace nnet{

template<typename Real>
Vector<Real>::Vector(int32 dim){
	dim_ = dim;
	data_ = (Real*)malloc(sizeof(Real)*dim_);

}

template<typename Real>
Vector<Real>::~Vector(){
	free(data_);
	data_ = NULL;
}

template<typename Real>
Real Vector<Real>::Sum(){
	Real sum = 0;
	for(int32 i = 0; i < dim_; i++)
		sum += data_[i];
	return sum;
}

template<typename Real>
Real Vector<Real>::Max(){
	Real max = data_[0];
	for(int32 i = 1; i < dim_; i++)
		if(data_[i] > max)
			max = data_[i];
	return max;
}

template<typename Real>
int32 Vector<Real>::MaxIndex(){
	int32 idex = 0;
	Real max = data_[0];
	for(int32 i = 1; i < dim_; i++){
		if(data_[i] > max)
		{
			max = data_[i];
			idex = i;
		}
	}
	return idex;
}

template<typename Real>
void Vector<Real>::Set(Real value){

	for(int32 i = 0; i < dim_; i++)
		data_[i] = value;
}

template<typename Real>
void Vector<Real>::Resize(int32 dim){

	dim_ = dim ;

	data_ = (Real*)realloc(data_, dim_*sizeof(Real));

	memset(data_, 0, sizeof(Real)*dim_) ;

}

template<typename Real>
void Vector<Real>::CopyFromPtr(Real* data, int32 length){

	assert( dim_ == length) ;

	free(data_) ;
	
	data_ = NULL ;
	
	data_ = (Real*)malloc(sizeof(Real)*dim_) ;

	for(int32 i = 0; i < dim_ ; i++ )
		data_[i] = data[i] ;
}

template<typename Real>
void Vector<Real>::SubVec(Vector<Real> &A){
	assert(A.Dim() == dim_);
	Real* data_a = A.Data();
	for(int32 i = 0; i < dim_; i++)
		data_[i] -= data_a[i];
}

template<typename Real>
void Vector<Real>::AddVec(Vector<Real> &A){
	assert(A.Dim() == dim_);
	Real* data_a = A.Data();
	for(int32 i = 0; i < dim_; i++)
		data_[i] += data_a[i];
}

template<typename Real>
void Vector<Real>::SumRowMat( Matrix<Real> &A, MatrixTransposeType TransA, Real alpha){
	if(TransA == kTrans)
		A.Transform();
	Real* data_a = A.Data();
	int32 Rows = A.NumRows();
	int32 Cols = A.NumCols();
	Vector<Real> S(Rows);
	S.Set(0.0);
	Real* data_sum = S.Data();
	for(int32 i = 0; i < Rows; i++)
		for(int32 j = 0; j < Cols; j++){
			data_sum[i] += data_a[i*Cols+j];  
		}

	this->AddVec(S);
}

template<typename Real>
void Vector<Real>::CopyFromvector(Vector<Real> &Vec){
	assert(Vec.Dim() == dim_);
	Real* data_vec = Vec.Data();
	for(int32 i = 0; i < dim_; i++){

		data_[i] = data_vec[i];
	}
}


//copy row from a cuMatrix of device
template<typename Real>
void Vector<Real>::CopyRowFromMat(Matrix<Real> &Mat, int32 row){

	assert(Mat.NumCols() == dim_);
	Real* data_mat = Mat.Data();
	for(int32 i = 0; i < dim_; i++)
		data_[i] = data_mat[(row-1)*dim_ + i];
}
	
//copy col from a cumatrix of device
template<typename Real>
void Vector<Real>::CopyColFromMat(Matrix<Real> &Mat, int32 col){
	assert(Mat.NumRows() == dim_);
	Real* data_mat = Mat.Data();

	for(int32 i = 0; i < dim_; i++)
		data_[i] = data_mat[i*(col) + col];
}

template<typename Real>
void Vector<Real>::Read(std::istream &is){
	for(int i = 0; i <dim_; i++)
		is >> data_[i];
}

template<typename Real>
void Vector<Real>::Write(std::ostream &os){
	for(int i = 0; i < dim_; i++)
		os << data_[i] ;
}

template class Vector<float>;

}//end namespace

