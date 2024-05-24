package org.yah.tools.cuda.api.driver;

import org.yah.tools.cuda.api.NativeEnum;

/**
 * Function properties
 */
public enum CUfunction_attribute implements NativeEnum {
    /**
     * The maximum number of threads per block, beyond which a launch of the
     * function would fail. This number depends on both the function and the
     * device on which the function is currently loaded.
     */
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,

    /**
     * The size in bytes of statically-allocated shared memory required by
     * this function. This does not include dynamically-allocated shared
     * memory requested by the user at runtime.
     */
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,

    /**
     * The size in bytes of user-allocated constant memory required by this
     * function.
     */
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,

    /**
     * The size in bytes of local memory used by each thread of this function.
     */
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,

    /**
     * The number of registers used by each thread of this function.
     */
    CU_FUNC_ATTRIBUTE_NUM_REGS,

    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for cubins
     * compiled prior to CUDA 3.0.
     */
    CU_FUNC_ATTRIBUTE_PTX_VERSION,

    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy cubins that do not have a
     * properly-encoded binary architecture version.
     */
    CU_FUNC_ATTRIBUTE_BINARY_VERSION,

    /**
     * The attribute to indicate whether the function has been compiled with
     * user specified option "-Xptxas --dlcm=ca" set .
     */
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,

    /**
     * The maximum size in bytes of dynamically-allocated shared memory that can be used by
     * this function. If the user-specified dynamic shared memory size is larger than this
     * value, the launch will fail.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,

    /**
     * On devices where the L1 cache and shared memory use the same hardware resources,
     * this sets the shared memory carveout preference, in percent of the total shared memory.
     * Refer to ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
     * This is only a hint, and the driver can choose a different ratio if required to execute the function.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,

    /**
     * If this attribute is set, the kernel must launch with a valid cluster
     * size specified.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET,

    /**
     * The required cluster width in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     * <p>
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,

    /**
     * The required cluster height in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     * <p>
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,

    /**
     * The required cluster depth in blocks. The values must either all be 0 or
     * all be positive. The validity of the cluster dimensions is otherwise
     * checked at launch time.
     * <p>
     * If the value is set during compile time, it cannot be set at runtime.
     * Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,

    /**
     * Whether the function can be launched with non-portable cluster size. 1 is
     * allowed, 0 is disallowed. A non-portable cluster size may only function
     * on the specific SKUs the program is tested on. The launch might fail if
     * the program is run on a different hardware platform.
     * <p>
     * CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking
     * whether the desired size can be launched on the current device.
     * <p>
     * Portable Cluster Size
     * <p>
     * A portable cluster size is guaranteed to be functional on all compute
     * capabilities higher than the target compute capability. The portable
     * cluster size for sm_90 is 8 blocks per cluster. This value may increase
     * for future compute capabilities.
     * <p>
     * The specific hardware unit may support higher cluster sizes that’s not
     * guaranteed to be portable.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,

    /**
     * The block scheduling policy of a function. The value type is
     * CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy.
     * See ::cuFuncSetAttribute, ::cuKernelSetAttribute
     */
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,

    CU_FUNC_ATTRIBUTE_MAX;

}
