// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef NNET_MATRIX_KERNEL_H_
#define NNET_MATRIX_KERNEL_H_

#include "../base/common.h"

extern"C"
{
	#include <cblas.h>
}

namespace nnet{


// cblas functions
// single precision
static void cblas_gemm(const enum CBLAS_ORDER Order , const enum CBLAS_TRANSPOSE TransA , const enum CBLAS_TRANSPOSE TransB , const int M , const int N , const int K , const float alpha , const float* A , const int lda , const float* B , const int ldb , const float beta , float* C , const int ldc)
{
	cblas_sgemm(Order , TransA , TransB , M , N , K , alpha , A , lda , B , ldb , beta , C , ldc) ;
}


static void cblas_scal(const int N , const float alpha , float* X , const int incX)
{
	cblas_sscal(N , alpha , X , incX) ;
}


static void cblas_axpy(const int n , const float alpha , const float* x , const int incx , float* y , const int incy)
{
	cblas_saxpy(n , alpha , x , incx , y , incy) ;
}


static float cblas_nrm2(const int N , const float* X , const int incX)
{
	return cblas_snrm2(N , X , incX) ;
}


//double precision
static void cblas_gemm(const enum CBLAS_ORDER Order , const enum CBLAS_TRANSPOSE TransA , const enum CBLAS_TRANSPOSE TransB , const int M , const int N , const int K , const double alpha , const double* A , const int lda , const double* B , const int ldb , const double beta , double* C , const int ldc)
{
	cblas_dgemm(Order , TransA , TransB , M , N , K , alpha , A , lda , B , ldb , beta , C , ldc) ;
}

static void cblas_scal(const int N , const double alpha , double* X , const int incX)
{
	cblas_dscal(N , alpha , X , incX) ;
}

static void cblas_axpy(const int n , const double alpha , const double* x , const int incx , double* y , const int incy)
{
	cblas_daxpy(n , alpha , x , incx , y , incy) ;
}

static double cblas_nrm2(const int N , const double* X , const int incX)
{
	return cblas_dnrm2(N , X , incX) ;
}

}//end namespace

#endif