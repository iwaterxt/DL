//cuMatrix/cuMatrix.h
// Copyright 2014-12-30  (Author: xutao)
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

#include "cuMatrix.h"
#include "cuKernel.h"
#include <curand.h>

namespace nnet{

template<typename Real>
cuMatrix<Real>::cuMatrix(){

	Rows_ = 0;
	Cols_ = 0;
	Stride_ = 0;
	data_ = NULL;
}

template<typename Real>
cuMatrix<Real>::cuMatrix(int32 Rows, int32 Cols)  {

	Rows_ = Rows;
	Cols_ = Cols;
	size_t pitch ;
	cudaMalloc((void**)&data_, sizeof(Real) * Rows * Cols);
            //cudaMallocPitch((void**)&data_, &pitch, Cols_ * sizeof(Real),  Rows_); 
            //Stride_ = pitch/sizeof(Real);
	Stride_ = Cols_ ;
	this->Set(0.0);
}

template<typename Real>
cuMatrix<Real>::~cuMatrix(){
	cuda_destory(data_);
	data_ = NULL;
}

template<typename Real>
void cuMatrix<Real>::FindRowMaxId(cuVector<Real> &Vec){

	assert(Rows_ == Vec.Dim()) ;
	cuda_FindRowMaxId(data_, Vec.Data(), Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::CopyFromMat(cuMatrix<Real> &Mat){
	//printf("the value of Rows_ is : %d, the value of Cols_ is : %d\n", Rows_, Cols_);
	//printf("the value of mat.row is: %d, the value of mat.cols is : %d\n", Mat.NumRows(), Mat.NumCols());
	assert(Rows_ == Mat.NumRows() && Cols_ == Mat.NumCols());
	//cudaMemcpy2D(data_ , Stride_ * sizeof(Real) , Mat.Data(), Mat.NumStride() * sizeof(Real), Mat.NumCols() * sizeof(Real), Rows_, cudaMemcpyDeviceToDevice);

	cudaMemcpy(data_, Mat.Data(), sizeof(Real)*Rows_*Cols_, cudaMemcpyDeviceToDevice);
	
}

template<typename Real>
void cuMatrix<Real>::CopyFromMat(Matrix<Real> &Mat){

	assert(Rows_ == Mat.NumRows() && Cols_ == Mat.NumCols());
	Real* data_mat = Mat.Data();
             
             //cudaMemcpy2D( data_, Stride_ * sizeof(Real), data_mat, Cols_* sizeof(Real), Cols_* sizeof(Real) , Rows_, cudaMemcpyHostToDevice ) ;
	cudaMemcpy(data_, data_mat, sizeof(Real)*Rows_*Cols_, cudaMemcpyHostToDevice);
}

template<typename Real>
void cuMatrix<Real>::CopyToMat(Matrix<Real> &Mat){
	assert(Rows_ == Mat.NumRows() && Cols_ == Mat.NumCols());
	Real* data_mat = Mat.Data();
	//cudaMemcpy2D(data_mat, sizeof(Real) * Mat.NumCols(), data_, Stride_ * sizeof(Real), Cols_ * sizeof(Real), Rows_, cudaMemcpyDeviceToHost);
	cudaMemcpy(data_mat, data_, sizeof(Real)*Rows_*Cols_, cudaMemcpyDeviceToHost);
}

template<typename Real>
void cuMatrix<Real>::CopyToHostPtr(Real* Ptr){
	//cudaMemcpy2D( Ptr, sizeof(Real) * Cols_, data_, Stride_ * sizeof(Real), Cols_ * sizeof(Real), Rows_, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ptr, data_, sizeof(Real)*Rows_*Cols_, cudaMemcpyDeviceToHost);
}


template<typename Real>
void cuMatrix<Real>::SumRowMatplusexp(cuVector<Real> &Vec){

	assert(Rows_ == Vec.Dim());

	this->Exp();

	Vec.SumRowMat(*this) ;

}

template<typename Real>
void cuMatrix<Real>::ApplySoftMaxPerRow(){

	cuVector<Real> Vec(Rows_);
	this->SumRowMatplusexp(Vec);
	Real* data_vec = Vec.Data();
	cuda_ApplySoftMaxPerRow(data_, data_vec, Cols_, Rows_, Stride_);
}

template<typename Real>
void cuMatrix<Real>::Sigmoid(cuMatrix<Real> &input){ 

	assert(Rows_ == input.NumRows() && Cols_ == input.NumCols());
	Real* data_mat = input.Data();
	cuda_Sigmoid(data_, data_mat, Cols_, Rows_, Stride_);
}

template<typename Real>
void cuMatrix<Real>::DiffSigmoid(cuMatrix<Real> &out, cuMatrix<Real> &out_diff){

	assert(Rows_ == out.NumRows() && out.NumRows() == out_diff.NumRows() && Cols_ == out.NumCols() && out.NumCols() == out_diff.NumCols());
	Real* data_out = out.Data();
	Real* data_out_diff = out_diff.Data();
	cuda_diffsigmoids(data_, data_out, data_out_diff, Cols_, Rows_, Stride_);
}

template<typename Real>
void cuMatrix<Real>::Tanh(cuMatrix<Real> &input){

	assert(Rows_ == input.NumRows() && Cols_ == input.NumCols());
	Real* data_mat = input.Data();
	cuda_Tanh(data_, data_mat, Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::DiffTanh(cuMatrix<Real> &out, cuMatrix<Real> &out_diff){

	assert(Rows_ == out.NumRows() && out.NumRows() == out_diff.NumRows() && Cols_ == out.NumCols() && out.NumCols() == out_diff.NumCols());
	Real* data_out = out.Data();
	Real* data_out_diff = out_diff.Data();
	cuda_difftanhs(data_, data_out, data_out_diff, Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::ReLU(cuMatrix<Real> &input){

	assert(Rows_ == input.NumRows() && Cols_ == input.NumCols());
	Real* data_mat = input.Data();
	cuda_ReLU(data_, data_mat, Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::DiffReLU(cuMatrix<Real> &out, cuMatrix<Real> &out_diff){

	assert(Rows_ == out_diff.NumRows() && Cols_ == out_diff.NumCols());
	Real* data_out = out.Data();
	Real* data_out_diff = out_diff.Data();
	cuda_diffrelus(data_, data_out, data_out_diff, Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::Set(Real value){
//setup the execution configuration

	Real* data_mat = (Real*)malloc(sizeof(Real)*Cols_*Rows_);

	for(int i = 0; i < Cols_*Rows_; i++)
		data_mat[i] = value ;

	//cudaMemcpy2D( data_, Stride_ * sizeof(Real), data_mat, Cols_* sizeof(Real), Cols_* sizeof(Real) , Rows_, cudaMemcpyHostToDevice ) ;
	cudaMemcpy(data_, data_mat, sizeof(Real)*Rows_*Cols_, cudaMemcpyHostToDevice);

	free(data_mat) ;
}

template<typename Real>
void cuMatrix<Real>::Log(){

	cuda_Log(data_, Cols_, Rows_, Stride_);
}

template<typename Real>
void cuMatrix<Real>::Exp(){

	cuda_Exp(data_, Cols_, Rows_, Stride_);
}

template<typename Real>
Real cuMatrix<Real>::Sum(){

	Real sum = 0.0;

	cuVector<Real> Vec_gpu(Rows_);

	Vec_gpu.SumRowMat(*this);

	sum = Vec_gpu.Sum();

	return sum ;
}

template<typename Real>
Real cuMatrix<Real>::AbSum(){

	Real sum = 0.0;

	cuVector<Real> Vec_gpu(Rows_);

	Vec_gpu.AbSumRowMat(*this);

	sum = Vec_gpu.Sum();

	return sum ;
}

template<typename Real>
void cuMatrix<Real>::Scale(Real value){

	cuda_scale(data_, Cols_, Rows_, Stride_, value);
}

template<typename Real>
void cuMatrix<Real>::Resize(int32 Rows, int32 Cols){

	if(data_ != NULL)
		cudaFree(data_);
	size_t pitch = 0 ;
	cudaMalloc((void**)&data_, sizeof(Real)*Cols*Rows);
	//cudaMallocPitch((void**)&data_, &pitch, Cols * sizeof(Real), Rows);
	Stride_ = Cols ; //pitch/sizeof(float);
	Rows_ = Rows ;
	Cols_ = Cols ;
	this->Set(0.0);

}

template<typename Real>
void cuMatrix<Real>::AddMatMat( cuMatrix<Real>&A, MatrixTransposeType transa,  cuMatrix<Real>&B,  MatrixTransposeType transb, Real alpha, Real beta){

	//C = alpha*op(A)*op(B) + beta*C

	int32 m = ((transb==kTrans)? B.NumRows() : B.NumCols()); 
    int32 n = ((transa==kTrans)? A.NumCols() : A.NumRows());
    int32 k = ((transb==kTrans)? B.NumCols() : B.NumRows());

	cublas_gemm( (transb==kTrans?'T':'N'), (transa==kTrans?'T':'N'), m , n , k , alpha , B.Data() , B.NumStride() , A.Data(), A.NumStride() , beta , data_ , Stride_);
	
}


template<typename Real>
void cuMatrix<Real>::BinarizeProbs(Real Probs){

	curandGenerator_t gen;
	float *p_d = NULL ;
	cudaMalloc((void **)&p_d, Cols_ * Rows_ * sizeof(float));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
	curandSetPseudoRandomGeneratorSeed(gen, 11ULL);
	curandGenerateUniform(gen, p_d, Rows_*Cols_);
	cuda_BinarizeProbs(data_, Cols_, Rows_, Probs, p_d);

}

template<typename Real>
void cuMatrix<Real>::ApplyFloor(Real value){

	cuda_ApplyFloor(data_, Cols_, Rows_, Stride_, value);
}

template<typename Real>
void cuMatrix<Real>::ApplyNorm(){
	cuda_ApplyNorm(data_, Cols_, Rows_, Stride_);
}

template<typename Real>
void cuMatrix<Real>::ApplyHeaviside(){

	cuda_ApplyHeaviside(data_, Cols_, Stride_, Rows_);
}


template<typename Real>
void cuMatrix<Real>::AddVecToRows(cuVector<Real> &Vec){
	assert(Vec.Dim() == Cols_) ;

	cuda_AddVecToRows(data_ , Vec.Data() , Cols_ , Rows_, Stride_) ;

}

template<typename Real>
void cuMatrix<Real>::MulElements(cuMatrix<Real> &Mat){
	assert(Rows_ == Mat.NumRows() && Cols_ == Mat.NumCols());
	Real* data_mat = Mat.Data();
	cuda_MulElements(data_, data_mat, Cols_, Rows_, Stride_);

}

template<typename Real>
void cuMatrix<Real>::AddMat( cuMatrix<Real> &Mat, MatrixTransposeType TransM, Real alpha){
	assert(Rows_ == Mat.NumRows() && Cols_ == Mat.NumCols());
	Real* data_mat = Mat.Data();
	cuda_addmat(data_, data_mat, Cols_, Rows_, Stride_, alpha);
}

template<typename Real>
void cuMatrix<Real>::Read(std::istream &is){

	Real* Ptr = (Real*)malloc(Rows_*Cols_*sizeof(Real)) ;
	ExpectToken(is , "[") ;
	for(int32 i = 0; i < Rows_*Cols_; i++)
		is >> Ptr[i];
	ExpectToken(is , "]") ;
	cudaMemcpy(data_, Ptr, sizeof(Real)*Rows_*Cols_, cudaMemcpyHostToDevice);
	//cudaMemcpy2D( data_, Stride_ * sizeof(Real), Ptr, Cols_* sizeof(Real), Cols_* sizeof(Real) , Rows_, cudaMemcpyHostToDevice ) ;
	if(Ptr != NULL)
		free(Ptr);
	Ptr = NULL;
}

template<typename Real>
void cuMatrix<Real>::Write(std::ostream &os){
	Real* Ptr = (Real*)malloc(Rows_*Cols_*sizeof(Real)) ;
	this->CopyToHostPtr(Ptr);
	os << "[ " ;
	for(int32 i = 0; i < Rows_*Cols_; i++){
		os << Ptr[i]<<" ";
	}
	os << "]" ;
	os <<'\n' ;
	if(Ptr != NULL)
		free(Ptr) ;
	Ptr = NULL ;
}

template<typename Real>
void cuMatrix<Real>::operator=(Matrix<Real> &Mat){


	assert(Mat.NumRows() == Rows_ && Mat.NumCols() == Cols_) ;
	Real* data_mat = Mat.Data() ;
	cudaMemcpy ( data_, Mat.Data(), sizeof(Real) * Cols_ * Rows_, cudaMemcpyHostToDevice);
	//cudaMemcpy2D( data_, Stride_ * sizeof(Real), data_mat, Cols_* sizeof(Real), Cols_* sizeof(Real) , Rows_, cudaMemcpyHostToDevice ) ;


}

template<typename Real>
void cuMatrix<Real>::operator=(cuMatrix<Real> &Mat){

	if(Mat.NumRows() == Rows_ && Mat.NumCols() == Cols_){

		//cudaMemcpy2D(data_ , Stride_ * sizeof(Real) , Mat.Data(), Mat.NumStride() * sizeof(Real), Mat.NumCols() * sizeof(Real), Rows_, cudaMemcpyDeviceToDevice);
		cudaMemcpy ( data_, Mat.Data(), sizeof(Real) * Cols_ * Rows_, cudaMemcpyDeviceToDevice);
	}
	else
	{
		if(NULL != data_) {
		    cudaFree(data_) ;
	                 data_ = NULL ;
		}
		Rows_ = Mat.NumRows();
		Cols_ = Mat.NumCols();
		//size_t pitch ;
            		//cudaMallocPitch((void**)&data_, &pitch, Cols_ * sizeof(Real),  Rows_); 
            		//Stride_ = pitch/sizeof(Real);
		cudaMalloc((void**)&data_, sizeof(Real)*Cols_*Rows_);
		Stride_ = Cols_ ;
		cudaMemcpy(data_, Mat.Data(), sizeof(Real)*Cols_*Rows_, cudaMemcpyHostToDevice);
		//cudaMemcpy2D(data_ , Stride_ * sizeof(Real) , Mat.Data(), Mat.NumStride() * sizeof(Real), Mat.NumCols() * sizeof(Real), Rows_, cudaMemcpyDeviceToDevice);
	}

}

template class cuMatrix<float>;


}//end of namespace
