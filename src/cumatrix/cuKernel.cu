//cuMatrix/cuKernel.h
// Copyright 2015-2-2   (Author: xutao)
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
# include <math.h>
# include "cuKernel-ansi.h"
# include <cuda_runtime.h>
# include <cublas.h>
# include <cuda.h>
# include <curand.h>
# include <stdio.h>


#define Num_Blocks  16
#define CUDA1DBLOCK 256
/**************vector kernel function**************/


__global__
static void _add_vec(float* x, float* y, int dim, float alpha){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < dim)
		x[i] += alpha * y[i];
}

__global__
static void _sumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < dim){
		x[i] = 0.0 ;
		for(int m = 0; m < col; m++)
			x[i] += y[i*stride + m];
	}
}

__global__
static void _absumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < dim){
		x[i] = 0.0 ;
		for(int m = 0; m < col; m++){

			if (y[i * stride + m] > 0.0)
			
			         x[i] += y[i*stride + m];
			else
			         x[i]  -= y[i*stride + m];
		}	
	}
}


__global__
static void _sumColMat_vec(float* x, float* y, int dim, int row, size_t stride){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	

	if (i < dim){
		x[i] = 0.0 ;
		for(int m = 0; m < row; m++)
			x[i] += y[m * stride + i];
	}

}

__global__
static void _max_vec(float* x, float* y, int dim){

	
	float tmp = 0.0;
	for(int j = 0; j < dim; j++)
		if(tmp < x[j])
			tmp = x[j];
	y[0] = tmp;

}

__global__
static void _maxindex_vec(float* x, float* y, int dim){
	
	float tmp = 0.0;
	int maxidx;
	for(int j = 0; j < dim; j++)
		if(tmp < x[j]){

			tmp = x[j];
			maxidx = j;
		}
	
	y[0] = (float)maxidx;

}

__global__
static void _sum_vec(float* x, float* y, int dim){
	
	y[0] = 0.0 ;
  	
  	for(int i = 0; i < dim; i++)
                   y[0] += x[i] ;
	
}

__global__
static void _exp_vec(float* x, int dim){
	
	int i = blockDim.x * blockIdx.x + threadIdx.x ;
	if(i < dim)
		x[i] = exp(x[i]) ;
}

__global__
static void _set_vec(float* x, int dim, float value){

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < dim)

		x[i] = value;
}

__global__
static void _scale_vec(float* x, int dim, float value){
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < dim)
		x[i] *= value ; 

}




/******************cumatrix kernel*************/
__global__
static void _FindRowMaxId(float* x, float* y, float* index, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if((i < cols) && (j < rows)){

		if(y[i] < x[i + j * stride]){

			y[i] = x[i  + j * stride] ;

			index[j] = i ;
		}
	}

}

                                                                                                                                                                                                                                                                                                                                                                                                                  
__global__
static void _ApplySoftMaxPerRow(float* x, float* y, int cols, int rows, size_t stride){

	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if(i < cols && j < rows)
		x[j * stride + i] = x[j * stride + i]/y[j];

}


__global__ 
static void _sigmoids(float* x, float* y, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){

		x[j*stride + i] = 1/(1+exp(-y[j*stride + i])); 
	}
}

__global__
static void _diffsigmoids(float* x, float* y, float* z, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){

		x[j*stride + i] = y[j*stride + i] * (1 - y[j*stride + i]) * z[j*stride + i];
	}
}

__global__
static void _relus(float* x, float* y, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){
		x[j*stride + i] = (y[j*stride+i]>0)? y[j*stride+i] : 0;
	}
}

__global__
static void _diffrelus(float* x, float* y, float* z , int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if((i < cols)&&(j < rows)){
		x[j * stride + i] = ((y[ j * stride + i]>0)? 1 : 0) * z[j * stride + i];
	}
}

__global__
static void _tanhs(float* x, float* y, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){
		x[j*stride + i] = (exp(y[j * stride + i]) - exp(-y[j * stride + i]))/(exp(y[j * stride + i]) + exp(-y[j * stride + i]));
	}
}

__global__
static void _difftanhs(float* x, float* y, float* z, int cols, int rows, size_t stride){
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){

		x[j * stride + i] = (1 - y[ j * stride + i] * y[j * stride + i] ) * z[j * stride + i];
	}
}

__global__
 void _Set(float* x, int cols, int rows, float value, size_t stride){

	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if((i < cols)&&(j < rows)){

		x[j * stride + i] =  value;
	}
}

__global__
static void _Log(float* x, int cols, int rows, size_t stride){

	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if((i < cols)&&(j < rows)){
		x[ j * stride + i] = log(x[j * stride + i]);
	}
}

__global__
static void _Exp(float* x, int cols, int rows, size_t stride){

	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if((i < cols)&&(j < rows)){
		x[j*stride + i] = exp(x[j*stride + i]);
	}
}

__global__
static void _scale(float* x, int cols, int rows, size_t stride, float value){
	
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if((i < cols)&&(j < rows)){
		x[j*stride + i] = x[j*stride + i] * value;
	}

}


 __global__
 static void _ApplyFloor(float* x, int cols, int rows, float value, size_t stride){
 	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;

	if((i < cols)&&(j < rows)){
		x[j * stride + i] =  (x[j * stride + i] >= value) ? x[j * stride + i] : 0;
	}
 }

 __global__
 static void _ApplyNorm(float* x, int cols, int rows, size_t stride){

  	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float max = 0.0;
	for(int m = 0; m < cols; m++){
		if(max < x[i * stride + m]) max = x[i * stride + m] ;
	}
	__syncthreads();

	for(int n = 0; n < cols; n++){

		x[i * stride + n] = x[ i * stride + n] / max ;
	}
 }

 __global__
 static void _ApplyHeaviside(float* x, int cols, int rows, size_t stride){

  	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;
	if((i < cols) && (j < rows)){
		x[j*stride + i] = (x[j* stride + i] > 0)? 1 : 0; 
	}
 }

__global__
static void _AddVecToRows(float* x, float * y, int cols, int rows, size_t stride){
	
	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;
	if((i < cols) && (j < rows))
		x[j*stride + i] = x[j*stride + i] + y[i];

}

__global__
static void _MulElements(float* x, float* y, int cols, int rows, size_t stride ){

 	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;
	if((i < cols) && (j < rows)){
		x[j*stride + i] = x[j*stride + i] * y[j*stride + i];
	}
}

__global__
static void _addmat(float* x, float* y, int cols, int rows, size_t stride, float alpha){

 	int j = blockIdx.y*blockDim.y + threadIdx.y ;
	int i = blockIdx.x*blockDim.x + threadIdx.x ;
	if((i < cols) && (j < rows)){
		x[j * stride + i] += alpha * y[j * stride + i];
	}
}


__global__
static void _BinarizeProbs(float* x, int cols, int rows, float* random){
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if((i < rows) && (j < cols)){

		x[i*cols + j] = random[i*cols + j]>0 ? 1.0 : 0.0 ;
	}
}

/********************************common*************************************/
void cudaF_destory(float* x){

	cudaFree(x);

}

/*********************************************************************************************************************/

void cudaF_FindRowMaxId(float* x, float* y, int cols, int rows, size_t stride) {
	
	float* index;
	float* show = new float[rows];
	for(int i = 0; i < rows ; i++)
		show[i] = 0.1 ;

	cudaMemcpy(y, show, sizeof(float)*rows, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&index, sizeof(float)*rows) ;
	dim3 dimBlock(Num_Blocks, Num_Blocks) ;
	dim3 dimGrid((cols + Num_Blocks -1)/Num_Blocks, (rows + Num_Blocks -1)/Num_Blocks) ;
	_FindRowMaxId<<<dimGrid, dimBlock>>>(x, y, index, cols, rows, stride) ;
	cudaMemcpy(y, index, sizeof(float)*rows, cudaMemcpyDeviceToDevice);
	delete [] show ;
	cudaFree(index);
}

void cudaF_ApplySoftMaxPerRow(float* x, float* y, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_ApplySoftMaxPerRow<<<dimGrid, dimBlock>>>(x, y, cols, rows, stride);
}

void cudaF_sigmoids(float* x, float* y, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_sigmoids<<<dimGrid, dimBlock>>> (x , y, cols, rows, stride);

}

void cudaF_diffsigmoids(float* x, float* y, float* z, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_diffsigmoids<<<dimGrid, dimBlock>>> (x , y, z, cols, rows, stride);
}

void cudaF_tanhs(float* x, float* y, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_tanhs<<<dimGrid, dimBlock>>> (x , y, cols, rows, stride);
}

void cudaF_difftanhs(float* x, float* y, float* z, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_difftanhs<<<dimGrid, dimBlock>>> (x , y, z, cols, rows, stride);
}

void cudaF_relus(float* x, float* y, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_relus<<<dimGrid, dimBlock>>> (x , y, cols, rows, stride);
}

void cudaF_diffrelus(float* x, float* y, float* z , int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x -1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_diffrelus<<<dimGrid, dimBlock>>> (x , y, z , cols, rows, stride);
}

void cudaF_Sets(float* x, int cols, int rows, float value, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	
	_Set<<<dimGrid, dimBlock>>> (x , cols, rows, value, stride);

}

void cudaF_Log(float* x, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_Log<<<dimGrid,dimBlock>>>(x, cols, rows, stride);
}

void cudaF_Exp(float* x, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_Exp<<<dimGrid,dimBlock>>>(x, cols, rows, stride);
}

void cudaF_scale(float* x, int cols, int rows, size_t stride, float value){
             dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_scale<<<dimGrid,dimBlock>>>(x, cols, rows, stride, value);
}



void cudaF_ApplyFloor(float* x,  int cols, int rows, float value, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_ApplyFloor<<<dimGrid, dimBlock>>> (x , cols, rows, value, stride);
}

void cudaF_ApplyNorm(float* x, int cols, int rows, size_t stride){
	dim3 dimGrid(Num_Blocks);
	dim3 dimBlock((rows + Num_Blocks -1) / Num_Blocks);
	_ApplyNorm<<<dimGrid, dimBlock>>> (x, cols, rows, stride);
}

void cudaF_ApplyHeaviside(float* x, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);
	_ApplyHeaviside<<<dimGrid, dimBlock>>>(x, cols, rows, stride);
}

void cudaF_AddVecToRows(float* x, float* y, int cols, int rows, size_t stride){
	
	dim3 dimBlock(Num_Blocks, Num_Blocks) ;
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y) ;

	_AddVecToRows<<<dimGrid, dimBlock>>>(x , y, cols , rows, stride) ;
}



void cudaF_MulElements(float* x, float* y, int cols, int rows, size_t stride){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);

	_MulElements<<<dimGrid, dimBlock>>> (x, y, cols, rows, stride);

}

void cudaF_addmat(float* x, float* y, int cols, int rows, size_t stride, float alpha ){

	dim3 dimBlock(Num_Blocks, Num_Blocks);
	dim3 dimGrid((cols + dimBlock.x - 1)/dimBlock.x , (rows + dimBlock.y - 1)/dimBlock.y);

	_addmat<<<dimGrid, dimBlock>>>(x, y, cols, rows, stride, alpha );

}


void cudaF_BinarizeProbs(float* x, int cols, int rows, float Probs, float* random){

	int blocksPerGrid = cols/Num_Blocks;

	if(cols % Num_Blocks) blocksPerGrid++;

	dim3 dimBlock(blocksPerGrid, blocksPerGrid);

	dim3 dimGrid(Num_Blocks, Num_Blocks);

	_BinarizeProbs<<<dimGrid, dimBlock>>> (x, cols, rows, random);
}


/*****************cuvector function**************/


//there should take care of the dim of vector!
void cudaF_add_vec(float* x, float* y, int dim, float alpha){


	dim3 dimGrid(Num_Blocks);
	dim3 dimBlock(ceil(dim/Num_Blocks)+1);
	_add_vec<<<dimGrid, dimBlock>>>(x, y, dim, alpha);

}

void cudaF_sumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){

	dim3 dimGrid(Num_Blocks);
	dim3 dimBlock((dim + Num_Blocks -1) / Num_Blocks);

	_sumRowMat_vec<<<dimGrid, dimBlock>>>(x, y, dim, col, stride);


}
 void cudaF_AbsumRowMat_vec(float* x, float* y, int dim, int col, size_t stride){
	dim3 dimGrid(Num_Blocks);
	dim3 dimBlock((dim + Num_Blocks -1) / Num_Blocks);

	_absumRowMat_vec<<<dimGrid, dimBlock>>>(x, y, dim, col, stride);

 }


void cudaF_sumColMat_vec(float* x, float* y, int dim, int row, size_t stride){

	dim3 dimGrid((dim+CUDA1DBLOCK-1)/CUDA1DBLOCK);
	dim3 dimBlock(CUDA1DBLOCK);


	_sumColMat_vec<<<dimGrid, dimBlock>>>(x, y, dim, row, stride);
}

void cudaF_max_vec(float* x, float* y, int dim){

	_max_vec<<<1,1>>>(x, y, dim);

}

void cudaF_maxindex_vec(float* x, float* y, int dim){

	_maxindex_vec<<<1,1>>>(x, y, dim);
}

void cudaF_sum_vec(float* x, float*y, int dim){

	_sum_vec<<<1,1>>>(x, y, dim);
}

void cudaF_exp_vec(float* x, int dim){
	
	dim3 dimGrid((dim + Num_Blocks*Num_Blocks - 1) / (Num_Blocks*Num_Blocks));
	dim3 dimBlock(Num_Blocks*Num_Blocks);
	_exp_vec<<<dimGrid, dimBlock>>> (x , dim);

}

void cudaF_set_vec(float* x, int dim, float value){

	dim3 dimGrid((dim + Num_Blocks*Num_Blocks - 1) / (Num_Blocks*Num_Blocks));
	dim3 dimBlock(Num_Blocks*Num_Blocks);
	_set_vec<<<dimGrid, dimBlock>>> (x, dim, value);
}

void cudaF_scale_vec(float* x, int dim, float value){
	dim3 dimGrid((dim + Num_Blocks*Num_Blocks - 1) / (Num_Blocks*Num_Blocks));
	dim3 dimBlock(Num_Blocks*Num_Blocks);
	_scale_vec<<<dimGrid, dimBlock>>> (x,dim,value);
}
