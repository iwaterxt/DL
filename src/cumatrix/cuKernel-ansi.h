//cuMatrix/cuKernel-ansi.h

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

#ifndef NNET_CUMATRIX_CUKERNEL_ANSI_H_

#define NNET_CUMATRIX_CUKERNEL_ANSI_H_



extern "C" {



 void cudaF_FindRowMaxId(float* x, float* y, int dim, int col, size_t stride) ;
 
 void cudaF_add_vec(float* x, float* y, int dim, float alpha);

 void cudaF_sumRowMat_vec(float* x, float* y, int dim, int col, size_t stride);

 void cudaF_AbsumRowMat_vec(float* x, float* y, int dim, int col, size_t stride);

 void cudaF_sumColMat_vec(float* x, float* y, int dim, int row, size_t stride);

 void cudaF_max_vec(float* x, float* y, int dim);

 void cudaF_maxindex_vec(float* x, float* y, int dim);

 void cudaF_sum_vec(float* x, float* y, int dim);

 void cudaF_set_vec(float* x, int dim, float value);

 void cudaF_exp_vec(float* x, int dim);

 void cudaF_scale_vec(float* x, int dim, float value);

 void cudaF_ApplySoftMaxPerRow(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_sigmoids(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_tanhs(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_relus(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_diffsigmoids(float* x, float* y, float* z, int cols, int rows, size_t stride);

 void cudaF_difftanhs(float* x, float* y, float* z, int cols, int rows, size_t stride);

 void cudaF_diffrelus(float* x, float* y, float* z , int cols, int rows, size_t stride);

 void cudaF_Sets(float* x , int cols, int rows, float value, size_t stride);

 void cudaF_Log(float* x, int cols, int rows, size_t stride);

 void cudaF_Exp(float* x, int cols, int rows, size_t stride);

 void cudaF_scale(float* x, int cols, int rows, size_t stride, float value);

 void cudaF_ApplyFloor(float* x , int cols, int rows, float value, size_t stride);

 void cudaF_ApplyNorm(float* x, int cols, int rows, size_t stride);
 
 void cudaF_ApplyHeaviside(float* x, int cols, int rows, size_t stride);

 void cudaF_AddVecToRows(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_MulElements(float* x, float* y, int cols, int rows, size_t stride);

 void cudaF_addmat(float* x, float* y, int cols, int rows, size_t stride, float alpha);

 void cudaF_Sclale(float* x, int cols, int rows, float value, size_t stride);

 void cudaF_destory(float* x);
 
 void cudaF_BinarizeProbs(float* x, int cols, int rows, float Probs, float* random);

}

#endif
