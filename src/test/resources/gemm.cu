typedef unsigned int uint32_t;

template<typename T>
__device__ void gemm(uint32_t M, T* A)
{
    printf("M=%d\n", M);
    for (int i=0;i<M;++i)
    {
        printf("%f", A[i]);
        if (i < M-1) printf(", ");
    }
    printf("];\n");
}

extern "C" __global__ void sgemm(uint32_t M, float* A)
{
    gemm<float>(M,A);
}

//extern "C" __global__ void dgemm(uint32_t M)
//{
//    gemm<double>(M);
//}
