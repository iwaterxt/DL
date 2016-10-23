//cuMatrix/cuMatrix.h
// Copyright 2015-1-8   (Author: xutao)
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

#include "cuVector.h"
#include "../base/common.h"
#include "cuKernel.h"

namespace nnet{

	template<typename Real>
	cuVector<Real>::~cuVector(){
		cuda_destory(data_);
		data_ = NULL;
	}

	template<typename Real>
	cuVector<Real>::cuVector(int32 dim){
		dim_ = dim ;
		cudaMalloc((void**)&data_, sizeof(Real)*dim);
		this->Set(0.0);
	}

	template <typename Real>
	void cuVector<Real>::Resize(int32 dim){

		dim_ = dim ;
		cudaFree(data_);
		cudaMalloc((void**)&data_, sizeof(Real)*dim);
		this->Set(0.0);
	}

	template <typename Real>
	void cuVector<Real>::AddVec(cuVector<Real> &Vec, Real alpha){
		
		assert(Vec.Dim() == dim_);
		Real* data_vec = Vec.Data();
		cuda_add_vec(data_, data_vec, dim_, alpha);//this function is impliment in cukernel.XX
	}

	template <typename Real>
	void cuVector<Real>::SumRowMat( cuMatrix<Real> &Mat){
		assert(Mat.NumRows() == dim_);

		int32 col = Mat.NumCols();
		int32 stride = Mat.NumStride();
		
		Real* data_mat = Mat.Data();

		cuda_sumRowMat_vec(data_, data_mat, dim_, col, stride);//this function is impliment in cukernel.XX
	}

	template <typename Real>
	void cuVector<Real>::AbSumRowMat( cuMatrix<Real> &Mat){
		assert(Mat.NumRows() == dim_);

		int32 col = Mat.NumCols();
		int32 stride = Mat.NumStride();
		
		Real* data_mat = Mat.Data();

		cuda_AbsumRowMat_vec(data_, data_mat, dim_, col, stride);//this function is impliment in cukernel.XX
	}

	template<typename Real>
	void cuVector<Real>::SumColMat( cuMatrix<Real> &Mat){
		assert(Mat.NumCols() == dim_);
		Real* data_mat = Mat.Data();
		int32 row = Mat.NumRows();
		int32 stride = Mat.NumStride();

		cuda_sumColMat_vec(data_, data_mat, dim_, row, stride);//this function is impliment in cukernel.XX
	}

	template <typename Real>
	Real cuVector<Real>::Max(){
		cuVector<Real> tmp(1);
		Real* data_tmp = tmp.Data();
		cuda_max_vec(data_, data_tmp, dim_);//this function is impliment in cukernel.xx
		return tmp(0);
	}

	template <typename Real>
	int32 cuVector<Real>::MaxIndex(){

		cuVector<Real> tmp(1);
		Real* data_tmp = tmp.Data();
		cuda_maxindex_vec(data_,data_tmp,dim_);
		return (int32)tmp(0);
	}

	template <typename Real>
	Real cuVector<Real>::Sum(){
		cuVector<Real> tmp(1);
		Real* data_tmp = tmp.Data();
		cuda_sum_vec(data_, data_tmp, dim_);//this function is impliment in cukernel.xx

		return tmp(0);
	}


	template <typename Real>
	void cuVector<Real>::Set(Real value){
		Real* vec = (Real*)malloc(sizeof(Real)*dim_);
		for(int32 i = 0 ; i < dim_ ; i++)
			vec[i] = value ;
		cudaMemcpy(data_, vec, sizeof(Real)*dim_, cudaMemcpyHostToDevice) ;
		free(vec) ;
	}

	template <typename Real>
	void cuVector<Real>::Exp(){

		cuda_exp_vec(data_, dim_);//this function is impliment in cukernel.xx
	}

	template <typename Real>
	void cuVector<Real>::Scale(Real value){
		cuda_scale_vec(data_, dim_, value);
	}

	template <typename Real>
	void cuVector<Real>::CopyFromVector(cuVector<Real> &Vec){
		Real * data_Vec = Vec.Data();
		assert(dim_ == Vec.Dim());

		cudaMemcpy(data_ , data_Vec , sizeof(Real) * dim_ , cudaMemcpyDeviceToDevice) ;
	}

	template <typename Real>
	void cuVector<Real>::CopyFromVector(Vector<Real> &Vec){

		assert(dim_ == Vec.Dim());
		Real* data_Vec = Vec.Data();

		cudaMemcpy(data_ , data_Vec , sizeof(Real) * dim_ , cudaMemcpyHostToDevice) ;
	}

	template <typename Real>
	void cuVector<Real>:: CopyToVector(Vector<Real> &Vec){
		
		assert(dim_ == Vec.Dim());
		Real* data_Vec = Vec.Data();

		cudaMemcpy(data_Vec , data_ , sizeof(Real) * dim_ , cudaMemcpyDeviceToHost) ;
	}

	template <typename Real>
	void cuVector<Real>::CopyFromMat(cuMatrix<Real> &Mat){

		Real* data_mat = Mat.Data();
		int32 row = Mat.NumRows();
		int32 col = Mat.NumCols();

		assert(dim_ == row*col);

		cudaMemcpy(data_ , data_mat , sizeof(Real) * dim_ , cudaMemcpyDeviceToDevice) ;
	}
	
	template<typename Real>
	void cuVector<Real>::Read(std::istream &is) {
		Real* Ptr = new Real[dim_] ;
		ExpectToken(is , "[") ;
		for(int32 i = 0; i < dim_; i++)
			is >> Ptr[i] ;
		ExpectToken(is , "]") ;
	
		cudaMemcpy(data_ , Ptr , sizeof(Real) * dim_ , cudaMemcpyHostToDevice) ;
		
		delete [] Ptr ;

		Ptr = NULL ;
	}

	template<typename Real>
	void cuVector<Real>::Write(std::ostream &os) {

		os<<"[ " ;

		Real* data_host = new Real[dim_] ;

		cudaMemcpy(data_host, data_, sizeof(Real)*dim_, cudaMemcpyDeviceToHost) ;

		for(int32 i = 0; i < dim_ ; i++)
			os<< data_host[i]<<" " ;

		os<<" ]" ;

		os<<'\n' ;

		delete [] data_host;

		data_host = NULL ;
	}
		
template class cuVector<float> ;
}//end of namespace
