#include "Vector.h"

int main(){

	using namespace nnet;

	Matrix<float> Mat(100,10);
	Mat.Set(2.0);

	Vector<float> V1(100);
	V1.Set(1.0);
	Vector<float> V2(100);
	V2.Set(2.0);
	V1.AddVec(V2);
	V1.SubVec(V2);
	printf("the sum of V1 is: %f\n", V1.Sum());
	V1.CopyColFromMat(Mat, 2);
	float sum = V1.Sum();
	printf("the sum of V1 is: %f\n", sum);


}
