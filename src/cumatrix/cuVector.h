//cuMatrix/cuMatrix.h
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
#ifndef CUMATRIX_CUVECTOR_H_
#define CUMATRIX_CUVECTOR_H_
//#include "../base/nnet-io.h"
#include "../base/common.h"
#include "cuMatrix.h"
#include "../matrix/Vector.h"
#include "cuKernel.h"
#include "../base/nnet-io.h"


namespace nnet{

template<typename Real> class cuMatrix;

template<typename Real> 
class cuVector{

public:

	cuVector(){}

	cuVector(int32 dim);

	~cuVector();

	inline int32 Dim(){return dim_;}

	inline Real* Data() {return data_;}

	void Resize(int32 dim_);

	// vector A plus vector B
	void AddVec(cuVector<Real> &Vec, Real alpha);

            // sum the row of matrix into a vector, dim = row
	void SumRowMat( cuMatrix<Real> &Mat );

	void AbSumRowMat( cuMatrix<Real> &Mat );

	// sum the col of matrix into a vector, dim = col
	void SumColMat( cuMatrix<Real> &Mat);

	// return the max value of vector
	Real Max();

	// return the index of max value, zero-based.
	int32 MaxIndex();

	// return the sum of the vector.
	Real Sum();

	//set every element of the vector to zero.
	void Set(Real value);

	void Exp() ;

	void Scale(Real value) ;

	//copy value from a std::vector of host
	void CopyFromVector(cuVector<Real> &Vec) ;

	//copy cuvector to vector
	void CopyFromVector(Vector<Real> &Vec) ;

	void CopyToVector(Vector<Real> &Vec) ;

	void CopyFromMat(cuMatrix<Real> &Mat) ;

	//copy row from a cuMatrix of device
	void CopyRowFromMat(cuMatrix<Real> &Mat, int32 Row);

	//copy col from a cumatrix of device
	void CopyColFromMat(cuMatrix<Real> &Mat, int32 Col);

	void Read(std::istream &is) ;

	void Write(std::ostream &os) ;

    Real operator() (int32 i) const{
		
		assert(i < dim_);
		Real value;
		cudaMemcpy(&value, data_+i, sizeof(Real), cudaMemcpyDeviceToHost);
		return value;
	}

	inline void operator=(Vector<Real> &Vec) {

		if(dim_ == Vec.Dim()){

			cudaMemcpy(data_, Vec.Data() , sizeof(Real)*dim_ , cudaMemcpyHostToDevice) ;

		}
		else{

			cudaFree(data_) ;
			data_ = NULL ;

			cudaMalloc((void**)& data_, Vec.Dim()) ;

			cudaMemcpy(data_, Vec.Data() , sizeof(Real)*dim_ , cudaMemcpyHostToDevice) ;

		}


	}

	inline void operator=(cuVector<Real> &Vec){

		if(dim_ == Vec.Dim()){

			cudaMemcpy(data_, Vec.Data() , sizeof(Real)*dim_ , cudaMemcpyDeviceToDevice) ;

		}
		else{

			cudaFree(data_) ;
			data_ = NULL ;

			cudaMalloc((void**)& data_, Vec.Dim()) ;

			cudaMemcpy(data_, Vec.Data() , sizeof(Real)*dim_ , cudaMemcpyDeviceToDevice) ;

		}

	}

private:

	int32 dim_ ;

	Real* data_ ;

	//Disallow_Copy_And_Assign(cuVector);
 };

}//end of namespace
#endif
