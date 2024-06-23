
/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParamsV2 {
    void* func;                     /**< Kernel to launch */
    #if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;                   /**< Grid dimensions */
        dim3 blockDim;                  /**< Block dimensions */
    #else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;                  /**< Grid dimensions */
        uint3 blockDim;                 /**< Block dimensions */
    #endif
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};
