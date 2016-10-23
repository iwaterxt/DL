// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef NNET_COMMON_H_
#define NNET_COMMON_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include <stdint.h>

namespace nnet {

typedef uint16_t        uint16;
typedef uint32_t        uint32;
typedef uint64_t        uint64;
typedef int16_t         int16;
typedef int32_t         int32;
typedef int64_t         int64;
typedef float           float32;
typedef double         double64;

#if (KALDI_DOUBLEPRECISION != 0)
typedef double  BaseFloat;
#else
typedef float   BaseFloat;
#endif
typedef enum {kTrans, kNoTrans} MatrixTransposeType;
typedef enum {kSetZero, kUndefined} DataConfigType;

#define CHECK_EQ(x, y, F, L) CHECK((x) == (y), F, L)

#define CHECK(x, F, L) \
	if(!(x)) \
		std::cout << "check failed!"<< F << L<<std::endl\
		<< ' '



}
#endif
