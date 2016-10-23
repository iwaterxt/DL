//cuMatrix/cuKernel.h
// Copyright 2014-12-24   (Author: xutao)
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
#ifndef NNET_CUMATRIX_CUKERNEL_H_
#define NNET_CUMATRIX_CUKERNEL_H_

# include <cuda_runtime.h>
# include <cublas.h>
# include <math.h>
# include "cuKernel-ansi.h"

namespace nnet{
//sigle precision
static void cublas_gemm(char transa , char transb , int m , int n , int k , float alpha , const float* A , int lda , const float* B , int ldb , float beta , float* C , int ldc)
{
	cublasSgemm(transa , transb , m , n , k , alpha , A , lda , B , ldb , beta , C , ldc) ;
}

static void cublas_scal(int n , float alpha , float* x , int incx)
{
	cublasSscal(n , alpha , x , incx) ;
}

static void cublas_axpy(int n , float alpha , const float* x , int incx , float* y , int incy)
{
	cublasSaxpy(n , alpha , x , incx , y , incy) ;
}

static float cublas_nrm2(int n , const float* x , int incx)
{
	return cublasSnrm2(n , x , incx) ;
}

static void cublas_copy(int n , const float* x , int incx , float* y , int incy)
{
	cublasScopy(n , x , incx , y , incy) ;
}
/*************************common**************/

static void cuda_destory(float* x){

	cudaF_destory(x);
	
}

/**********Vector kernel function************/


static void cuda_add_vec(float* x, float* y, int dim, float alpha){

	cudaF_add_vec(x, y, dim, alpha);
}

static void cuda_sumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){
	cudaF_sumRowMat_vec(x, y, dim, col, stride);
}

static void cuda_AbsumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){
	cudaF_AbsumRowMat_vec(x,y,dim,col,stride);
}

static void cuda_sumColMat_vec(float* x, float* y, int dim, int row, size_t stride){
	cudaF_sumColMat_vec(x, y, dim, row, stride);
}

static void cuda_max_vec(float* x, float* y, int dim){

	
	cudaF_max_vec(x, y, dim);
	
}

static void cuda_maxindex_vec(float* x, float* y, int dim){
	
	cudaF_maxindex_vec(x, y, dim);
	
}

static void cuda_sum_vec(float* x, float* y, int dim){
	
	cudaF_sum_vec(x, y, dim);
	
}

static void cuda_exp_vec(float* x, int dim){

	cudaF_exp_vec(x , dim);
}

static void cuda_set_vec(float* x, int dim, float value){
	cudaF_set_vec(x, dim, value);
}

static void cuda_scale_vec(float* x, int dim, float value){
	cudaF_scale_vec(x, dim, value);
}


/********************cumatrix kernel*************/

static void cuda_FindRowMaxId(float* x, float* y, int cols, int rows, size_t stride){

	cudaF_FindRowMaxId( x,  y, cols, rows, stride) ;
}


static void cuda_ApplySoftMaxPerRow(float* x, float* y, int cols, int rows, size_t stride){

	cudaF_ApplySoftMaxPerRow(x, y,  cols, rows, stride);
}

static void cuda_Sigmoid(float* x, float* y, int cols, int rows, size_t stride){
	cudaF_sigmoids(x, y, cols, rows, stride);
} 

static void cuda_Tanh(float* x, float* y, int cols, int rows, size_t stride){
	cudaF_tanhs(x, y, cols, rows, stride);
}

static void cuda_ReLU(float* x, float* y, int cols, int rows, size_t stride){
	cudaF_relus(x, y, cols, rows, stride);
}

static void cuda_diffsigmoids(float* x, float* y, float* z, int cols, int rows, size_t stride){
	cudaF_diffsigmoids(x, y, z, cols, rows, stride);
}

static void cuda_difftanhs(float* x, float* y, float* z, int cols, int rows, size_t stride){
	cudaF_difftanhs(x, y, z, cols, rows, stride);
}

static void cuda_diffrelus(float* x, float* y, float* z, int cols, int rows, size_t stride){
	cudaF_diffrelus(x, y, z, cols, rows, stride);
}

static void cuda_Set(float* x , int cols, int rows, float value, size_t stride){

	cudaF_Sets(x, cols, rows, value, stride);
}

static void cuda_Log(float* x, int cols, int rows, size_t stride){

	cudaF_Log(x, cols, rows, stride);
}

static void cuda_Exp(float* x, int cols, int rows, size_t stride){

	cudaF_Exp(x, cols, rows, stride);
}


static void cuda_ApplyFloor(float* x , int cols, int rows, float value, size_t stride){
	cudaF_ApplyFloor(x , cols, rows, value, stride);
}

static void cuda_ApplyNorm(float* x, int cols, int rows, size_t stride){
	cudaF_ApplyNorm(x, cols, rows, stride);
}

static void cuda_ApplyHeaviside(float* x, int cols, int rows, size_t stride){
	cudaF_ApplyHeaviside(x, cols, rows, stride);
}

static void cuda_AddVecToRows(float* x, float* y, int cols, int rows, size_t stride){
	cudaF_AddVecToRows(x , y, cols, rows, stride) ;
}

static void cuda_MulElements(float* x, float* y, int cols, int rows, size_t stride ){
	cudaF_MulElements(x, y, cols, rows, stride);
}

static void cuda_addmat(float* x, float* y, int cols, int rows, size_t stride, float alpha){
	cudaF_addmat(x, y, cols, rows, stride, alpha);
}

static void cuda_scale(float* x, int cols, int rows, size_t stride, float value){

	cudaF_scale(x , cols, rows, stride, value);
}


static void cuda_BinarizeProbs(float* x, int cols, int rows, float Probs, float* random){

	cudaF_BinarizeProbs(x, cols, rows, Probs, random);

}

/**********************************************/

//double precision

static void cublas_gemm(char transa , char transb , int m , int n , int k , float alpha , const double* A , int lda , const double* B , int ldb , float beta , double* C , int ldc)
{
	cublasDgemm(transa , transb , m , n , k , alpha , A , lda , B , ldb , beta , C , ldc) ;
}

static void cublas_scal(int n , double alpha , double* x , int incx)
{
	cublasDscal(n , alpha , x , incx) ;
}

static void cublas_axpy(int n , double alpha , const double* x , int incx , double* y , int incy)
{
	cublasDaxpy(n , alpha , x , incx , y , incy) ;
}

static double cublas_nrm2(int n , const double* x , int incx)
{
	return cublasDnrm2(n , x , incx) ;
}

static void cublas_copy(int n , const double* x , int incx , double* y , int incy)
{
	cublasDcopy(n , x , incx , y , incy) ;
}

}//end of namespace

#endif