#include "cuVector.h"
#include "cuMatrix.h"
#include "../matrix/Matrix.h"

int main(){

	using namespace nnet;
	
	
	int m = 4 ;
	int n = 2 ;
	int k = 1 ;
	
	float* mat1 = (float*)malloc(sizeof(float)*m*n) ;
	float* mat2 = (float*)malloc(sizeof(float)*n*k) ;
	float* mat3 = (float*)malloc(sizeof(float)*m*k) ;

	cuMatrix<float> dev1(m,n);
	cuMatrix<float> dev2(n,k);
	cuMatrix<float> dev3(m,k);
	cuVector<float> dev(k);
	dev3.Scale(-1.0);
             cudaError_t err = cudaGetLastError();
	printf("%s, %d\n", cudaGetErrorString(err),__LINE__);  
	dev1.Set(1.0) ;
	dev2.Set(-1.0) ;
	dev3.Set(-1.0) ;
	dev.Set(1.0) ;
	cudaMemcpy2D(mat3, k* sizeof(float), dev3.Data(), dev3.NumStride()* sizeof(float), k * sizeof(float),  m, cudaMemcpyDeviceToHost);

             err = cudaGetLastError();
             
	printf("%s, %d\n", cudaGetErrorString(err),__LINE__);  
 	
	printf("the sum of dev3 is: %f, %s\n", dev3.AbSum(), __FILE__);
	printf("the sum of dev1 is: %f, %s\n", dev1.Sum(), __FILE__);
	printf("the sum of dev2 is: %f, %s\n", dev2.Sum(), __FILE__);

	printf("the stride of dev1 is %d\n", dev1.NumStride());

	dev3.AddMatMat(dev1, kNoTrans, dev2, kNoTrans, 1.0, 0.0);

	cudaMemcpy2D(mat3, k* sizeof(float), dev3.Data(), dev3.NumStride()* sizeof(float), k * sizeof(float),  m, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m; i++)
		for(int j = 0; j < k; j++)
			printf("%f  ", mat3[i*k+j]);
		dev3.Scale(-1.0);
	printf("the sum of dev3 is: %f, %s\n", dev3.Sum(), __FILE__);

	printf("the sum of dev is: %f, %s\n", dev.Sum(), __FILE__);

	dev.SumColMat(dev3);

	printf("the sum of dev is: %f, %s\n", dev.Sum(), __FILE__);


	err = cudaGetLastError();
	printf("%s, %d\n",cudaGetErrorString(err),__LINE__);

	//std::cout << "the row of Matrix is: "<<Matrix.NumRows()<<" the col of Matrix is: "<<Matrix.NumCols()<<std::endl;
	
	return 0 ;

 }
