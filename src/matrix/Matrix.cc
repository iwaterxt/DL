//Matrix/Matrix.cc
// Copyright 2015-1-24   (Author: xutao)
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

#include "Matrix.h"
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>


namespace nnet{

template<typename Real>
Matrix<Real>::Matrix(int32 Rows, int32 Cols){
	Rows_ = Rows;
	Cols_ = Cols;
	data_ = (Real*)malloc(sizeof(Real)*Rows*Cols);
}

template<typename Real>
Matrix<Real>::~Matrix(){
	free(data_);
	data_ = NULL;
}

template<typename Real>
Matrix<Real>& Matrix<Real>::operator=( Matrix &rhs){
	if(this != &rhs){
		free(data_);
	    data_ = rhs.Data();
	    Rows_ = rhs.NumRows();
	    Cols_ = rhs.NumCols();
	}
	return *this;
}

template<typename Real>
void Matrix<Real>::CopyFromPtr(Real* data, int32 length){

	if(this->Data() != data){

		assert(length == Rows_*Cols_) ;

		free(data_);

		data_ = (Real*)malloc(sizeof(Real)*length) ;

		for(int32 i = 0; i < length ; i++)
			data_[i] = data[i] ;
	}

}
//change vector into matrix
template<typename Real>
void Matrix<Real>::Vector2Matrix(Vector<Real> &Vector) {

	this->Set(0.0);

	assert(Rows_ == Vector.Dim()) ;

	for(int32 i = 0; i < Rows_ ; i++){

		data_[i*Cols_ + int32(Vector(i))] = 1.0 ;
	}


}

template<typename Real>
bool Matrix<Real>::SamDim( Matrix<Real> &A, MatrixTransposeType TransA){
	if(TransA == kTrans)
		A.Transform();
	assert(Rows_ == A.NumRows()&&Cols_ == A.NumCols());
	return true;
}

template<typename Real>
Real Matrix<Real>::Sum(){

	Real sum = 0;
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			sum += *(data_+i*Cols_+j);
		}
	return sum;
}

template<typename Real>
void Matrix<Real>::Set(Real value){
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++)
			*(data_+i*Cols_ + j) = value;
}

template<typename Real>
void Matrix<Real>::Resize(int32 Rows, int32 Cols){
	if((Rows_ == Rows)&&(Cols_ == Cols)){
	  for(int32 i = 0; i < Rows; i++)
		for(int32 j = 0; j < Cols; j++)
			*(data_+i*Cols + j) = 0;
	}
	else
	{
		free(data_);
		data_ = (Real*)malloc(sizeof(Real)*Rows*Cols);
	  	for(int32 i = 0; i < Rows; i++)
			for(int32 j = 0; j < Cols; j++)
				*(data_+i*Cols + j) = 0;
	}
}

template<typename Real>
void Matrix<Real>::Transform(){
	
	Real* data = (Real*)malloc(sizeof(Real)*Rows_*Cols_);

	for(int32 i = 0; i < Rows_; i++){
		for(int32 j = 0; j < Cols_; j++){

			*(data+ i*Cols_ + j) = *(data_+ j*Cols_ +i);
		}
	}

	int32 temp = Rows_;

	Rows_ = Cols_;

	Cols_ = temp;

	free(data_);

	data_ = data;
}


template<typename Real>
void Matrix<Real>::Normalized_Cmvn(){

   int32 num_rows = Rows_, num_cols = Cols_;
   Real* norm = (Real*)malloc(num_cols * sizeof(Real)) ;
   Real* norm_pow = (Real*)malloc(num_cols * sizeof(Real));
   Real* var = (Real*)malloc(num_cols * sizeof(Real)) ;
   Real* sum = (Real*)malloc(num_cols * sizeof(Real));
   Real* sum_squre = (Real*)malloc(num_cols * sizeof(Real));

   for(int32 i = 0; i < num_cols ; i++){
        sum[i] = 0.0 ;
        sum_squre[i] = 0.0 ;
        norm[i] = 0.0 ;
        norm_pow[i] = 0.0 ;
        var[i] = 0.0 ;
   }
   for(int32 i = 0; i < num_cols ; i++){
        for(int32 j = 0; j < num_rows ; j++){
            sum[i] += (*this)(j,i) ;
            sum_squre[i] += pow((*this)(j,i),2);
        }
   }
   for(int32 m = 0; m < num_cols; m++){

        norm[m] = sum[m] / num_rows ;
        norm_pow[m] = sum_squre[m] / num_rows ;

        var[m] = norm_pow[m] - pow(norm[m], 2);
        //printf("%f ", var[m]);
        if(var[m] < 1.0e-20) var[m] = 1.0e-20 ;
        //switch to shift and scale
        var[m] = 1.0/sqrt(var[m]);
        norm[m] *= (-1.0) * var[m] ;

   }

   for(int32 i = 0; i < num_cols ; i++)
        for(int32 j = 0; j < num_rows ; j++){

        (*this)(j,i) = (*this)(j,i)*var[i] + norm[i] ;  
   }

   free(norm);
   free(norm_pow);
   free(var);
   free(sum);
   free(sum_squre);
}

template<typename Real>
void Matrix<Real>::Random(Real param_stddev){

	for (int32 r=0; r<Rows_; r++) {

     	for (int32 c=0; c<Cols_; c++) {

       	 	(*this)(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
                   
      	}
    }
}


template<typename Real>
void Matrix<Real>::AddMatMat( Matrix<Real> &A, MatrixTransposeType TransA,  Matrix<Real> &B, MatrixTransposeType TransB, Real alpha, Real beta){
	int32 m = ((TransB==kTrans)? B.NumRows() : B.NumCols()); 
    int32 n = ((TransA==kTrans)? A.NumCols() : A.NumRows());
    int32 k = ((TransB==kTrans)? B.NumCols() : B.NumRows());
    int32 k1 = ((TransA==kTrans)? A.NumRows() : A.NumCols());

    assert(m == NumCols());
    assert(n == NumRows());
    assert(k == k1);
    if(TransA == kTrans) A.Transform();
    if(TransB == kTrans) B.Transform();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A.Data(), A.NumCols(),B.Data(), B.NumCols(),beta, data_, Cols_);
}
//f(x) = 1/(1+exp(-x))
template<typename Real>
void Matrix<Real>::Sigmoid(Matrix<Real> &A, MatrixTransposeType TransA){
	if(TransA == kTrans) A.Transform();
	Real* data_a = A.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++)
		{
			data_[i*Cols_+j] = 1.0 /(1.0 + exp(-data_a[i*Cols_+j])) ;
		}
}

//f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
template<typename Real>
void Matrix<Real>::Tanh(Matrix<Real> &A, MatrixTransposeType TransA){
	if(TransA == kTrans) A.Transform();
	Real* data_a = A.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] = (exp(data_a[i*Cols_+j])-exp(data_a[i*Cols_+j]))/(exp(data_a[i*Cols_+j])-exp(data_a[i*Cols_+j]));
		}	
}

//y = x>0? x:0;
template<typename Real>
void Matrix<Real>::ReLU(Matrix<Real> &A, MatrixTransposeType TransA){
	if(TransA == kTrans) A.Transform();
	Real* data_a = A.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] = data_a[i*Cols_+j]>0? data_a[i*Cols_+j] : 0;
		}
}

//ey/ex = y(1-y) && ez/ex = (ez/ey)*(ey/ex)
template<typename Real>
void Matrix<Real>::DiffSigmoid( Matrix<Real> &out,  Matrix<Real> &out_diff){
	assert((Rows_ == out.NumRows()) &&(Rows_ == out_diff.NumRows()));
	assert((Cols_ == out.NumCols()) &&(Cols_ == out_diff.NumCols()));

	Real* data = out.Data(); 
	Real* diff_data = out_diff.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] = diff_data[i*Cols_+j]*data[i*Cols_+j]*(1.0 - data[i*Cols_ + j]);
		}
}
//ey/ex = (1-y^2)
template<typename Real>
void Matrix<Real>::DiffTanh( Matrix<Real> &out,  Matrix<Real> &out_diff){
	assert((Rows_ == out.NumRows()) &&(Rows_ == out_diff.NumRows()));
	assert((Cols_ == out.NumCols()) &&(Cols_ == out_diff.NumCols()));

	Real* data = out.Data();
	Real* diff_data = out_diff.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] = diff_data[i*Cols_+j] * (1-data[i*Cols_+j]*data[i*Cols_+j]);
		}
}

//ey/ex = 1 for x>0
template<typename Real>
void Matrix<Real>::DiffReLU( Matrix<Real> &out,  Matrix<Real> &out_diff){
	assert((Rows_ == out.NumRows()) &&(Rows_ == out_diff.NumRows()));
	assert((Cols_ == out.NumCols()) &&(Cols_ == out_diff.NumCols()));
	Real* data = out.Data();
	Real* diff_data = out_diff.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] = diff_data[i*Cols_+j] * data[i*Cols_+j];
		}
}

template<typename Real>

void Matrix<Real>::SubMat( Matrix<Real> &A, MatrixTransposeType TransA){
	if(TransA == kTrans) A.Transform();
	assert(SamDim(A,kNoTrans));
	Real* data = A.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){

			data_[i*Cols_+j] -=  data[i*Cols_+j];
		}
}

template<typename Real>
void Matrix<Real>::AddMat( Matrix<Real> &A, MatrixTransposeType TransA, Real alpha){
	if(TransA == kTrans)
		A.Transform();
	assert(Rows_ == A.NumRows()&&Cols_==A.NumCols());
	Real* data = A.Data();
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			data_[i*Cols_+j] += alpha*data[i*Cols_+j];
		}
 }


template<typename Real>
void Matrix<Real>::BinarizeProbs(Real value){
	srand((int32)time(NULL));
	for(int32 i = 0; i < Rows_; i++)
		for(int32 j = 0; j < Cols_; j++){
			if(rand()%100 > 100.0*value)
				data_[i*Cols_ + j] = 0;
			else
				data_[i*Cols_ + j] = 1;		
		}
}

template<typename Real>
void Matrix<Real>::Read(std::istream &is){
	for(int i = 0; i <Rows_*Cols_; i++){
		is >> data_[i];
	}
}

template<typename Real>
void Matrix<Real>::Write(std::ostream &os){
	for(int i = 0; i < Rows_*Cols_; i++)
		os << data_[i] ;
}



template class Matrix<float>;
//template class Matrix<double>;

}//end