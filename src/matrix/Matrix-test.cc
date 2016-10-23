
#include "Matrix.h"



int main(){

	using namespace nnet;

	Matrix<float> A(128,128);
	Matrix<float> B(128,128);
	Matrix<float> C(128,128);

    if(A.SamDim(B,kNoTrans));
    	printf("A and B are the same dim\n");
	A.Set(1.0);
	B.Set(1.0);
	C.Random(2.0);

	C.Normalized_Cmvn();



	//A.AddMatMat( B, kNoTrans,  C, kNoTrans, 1.0, 1.0);

	float sum1 = C.Sum();


    printf("the sum of matrix C is : %f\n", sum1);

    //A.BinarizeProbs(0.5);

    float sum2 = A.Sum();

    printf("the sum of matrix A is : %f\n", sum2);

    A.AddMat(B,kNoTrans, 0.5);

    float sum3 = A.Sum();

    printf("the sum of matrix A is : %f\n", sum3);


}

