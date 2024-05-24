#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

namespace sandbox {
    namespace TestKernel {
        typedef struct KernelTestStruct {
            int value;
        } KernelTestStruct;
        __constant__ KernelTestStruct structField;
         __global__ void add(int N, double* a, double* b, double* c);
    }
}
