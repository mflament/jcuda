#define N 1000

extern "C" {
    __global__ void matSum(int* a, int* b, int* c)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x)
        {
            c[i] = a[i] + b[i];
        }
    }

    __global__ void helloCuda()
    {
//         printf("blockIdx(%d,%d,%d) threadIdx(%d,%d,%d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    }
}