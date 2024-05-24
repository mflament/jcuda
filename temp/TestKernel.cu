#include "TestKernel.cuh"

namespace sandbox {

    namespace TestKernel {
        __global__ void add(int N, double* a, double* b, double* c, KernelTestStruct* testStructure) {
            int stride = blockDim.x * gridDim.x;
            printf("structField.value=%d\n", structField.value);
            for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride) {
                c[i] = a[i] + b[i] + testStructure->value;
            }
        }
    }

}
