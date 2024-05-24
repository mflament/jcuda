typedef unsigned long long uint64_t;

namespace sandbox {

    namespace TestKernel {
        typedef struct KernelTestStruct {
            double value;
        } KernelTestStruct;

        __constant__ KernelTestStruct structField;

        __global__ void test(int N) {
//             if (N < 0)
//                 printf("%d is negative\n", N);
//             else
//                 printf("%d is positive\n", N);
        }
    }

}
