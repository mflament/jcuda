package org.yah.tools.cuda.api.driver;

import org.yah.tools.cuda.api.NativeEnum;

public enum CUjit_option implements NativeEnum {
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    CU_JIT_MAX_REGISTERS,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization of the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Cannot be combined with ::CU_JIT_TARGET.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    CU_JIT_THREADS_PER_BLOCK,

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     */
    CU_JIT_WALL_TIME,

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    CU_JIT_INFO_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    CU_JIT_ERROR_LOG_BUFFER,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    CU_JIT_OPTIMIZATION_LEVEL,

    /**
     * No option value required. Determines the target based on the current
     * attached context (default)\n
     * Option type: No option value needed\n
     * Applies to: compiler and linker
     */
    CU_JIT_TARGET_FROM_CUCONTEXT,

    /**
     * Target is chosen based on supplied ::CUjit_target.  Cannot be
     * combined with ::CU_JIT_THREADS_PER_BLOCK.\n
     * Option type: unsigned int for enumerated type ::CUjit_target\n
     * Applies to: compiler and linker
     */
    CU_JIT_TARGET,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::CUjit_fallback.  This option cannot be
     * used with cuLink* APIs as the linker requires exact matches.\n
     * Option type: unsigned int for enumerated type ::CUjit_fallback\n
     * Applies to: compiler only
     */
    CU_JIT_FALLBACK_STRATEGY,

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    CU_JIT_GENERATE_DEBUG_INFO,

    /**
     * Generate verbose log messages (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    CU_JIT_LOG_VERBOSE,

    /**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    CU_JIT_GENERATE_LINE_INFO,

    /**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::CUjit_cacheMode_enum.\n
     * Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum\n
     * Applies to: compiler only
     */
    CU_JIT_CACHE_MODE,

    /**
     * \deprecated
     * This jit option is deprecated and should not be used.
     */
    CU_JIT_NEW_SM3X_OPT,

    /**
     * This jit option is used for internal purpose only.
     */
    CU_JIT_FAST_COMPILE,

    /**
     * Array of device symbol names that will be relocated to the corresponding
     * host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * When loading a device module, driver will relocate all encountered
     * unresolved symbols to the host addresses.\n
     * It is only allowed to register symbols that correspond to unresolved
     * global variables.\n
     * It is illegal to register the same device symbol at multiple addresses.\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     */
    CU_JIT_GLOBAL_SYMBOL_NAMES,

    /**
     * Array of host addresses that will be used to relocate corresponding
     * device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.\n
     * Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.\n
     * Option type: void **\n
     * Applies to: dynamic linker only
     */
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES,

    /**
     * Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
     * ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     */
    CU_JIT_GLOBAL_SYMBOL_COUNT,

    /**
     * \deprecated
     * Enable link-time optimization (-dlto) for device code (Disabled by default).\n
     * This option is not supported on 32-bit platforms.\n
     * Option type: int\n
     * Applies to: compiler and linker
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_LTO,

    /**
     * \deprecated
     * Control single-precision denormals (-ftz) support (0: false, default).
     * 1 : flushes denormal values to zero
     * 0 : preserves denormal values
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_FTZ,

    /**
     * \deprecated
     * Control single-precision floating-point division and reciprocals
     * (-prec-div) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_PREC_DIV,

    /**
     * \deprecated
     * Control single-precision floating-point square root
     * (-prec-sqrt) support (1: true, default).
     * 1 : Enables the IEEE round-to-nearest mode
     * 0 : Enables the fast approximation mode
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_PREC_SQRT,

    /**
     * \deprecated
     * Enable/Disable the contraction of floating-point multiplies
     * and adds/subtracts into floating-point multiply-add (-fma)
     * operations (1: Enable, default; 0: Disable).
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_FMA,

    /**
     * \deprecated
     * Array of kernel names that should be preserved at link time while others
     * can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.\n
     * Note that kernel names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all kernels with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_REFERENCED_KERNEL_NAMES,

    /**
     * \deprecated
     * Number of entries in ::CU_JIT_REFERENCED_KERNEL_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: dynamic linker only
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_REFERENCED_KERNEL_COUNT,

    /**
     * \deprecated
     * Array of variable names (__device__ and/or __constant__) that should be
     * preserved at link time while others can be removed.\n
     * Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.\n
     * Note that variable names can be mangled by the compiler in which case the
     * mangled name needs to be specified.\n
     * Wildcard "*" can be used to represent zero or more characters instead of
     * specifying the full or mangled name.\n
     * It is important to note that the wildcard "*" is also added implicitly.
     * For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
     * thus preserve all variables with those names. This can be avoided by providing
     * a more specific name like "barfoobaz".\n
     * Option type: const char **\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_REFERENCED_VARIABLE_NAMES,

    /**
     * \deprecated
     * Number of entries in ::CU_JIT_REFERENCED_VARIABLE_NAMES array.\n
     * Option type: unsigned int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_REFERENCED_VARIABLE_COUNT,

    /**
     * \deprecated
     * This option serves as a hint to enable the JIT compiler/linker
     * to remove constant (__constant__) and device (__device__) variables
     * unreferenced in device code (Disabled by default).\n
     * Note that host references to constant and device variables using APIs like
     * ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
     * the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.\n
     * Option type: int\n
     * Applies to: link-time optimization specified with CU_JIT_LTO
     *
     * Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
     */
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES,

    /**
     * Generate position independent code (0: false)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    CU_JIT_POSITION_INDEPENDENT_CODE,

    /**
     * This option hints to the JIT compiler the minimum number of CTAs from the
     * kernel’s grid to be mapped to a SM. This option is ignored when used together
     * with ::CU_JIT_MAX_REGISTERS or ::CU_JIT_THREADS_PER_BLOCK.
     * Optimizations based on this option need ::CU_JIT_MAX_THREADS_PER_BLOCK to
     * be specified as well. For kernels already using PTX directive .minnctapersm,
     * this option will be ignored by default. Use ::CU_JIT_OVERRIDE_DIRECTIVE_VALUES
     * to let this option take precedence over the PTX directive.
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    CU_JIT_MIN_CTA_PER_SM,

    /**
     * Maximum number threads in a thread block, computed as the product of
     * the maximum extent specifed for each dimension of the block. This limit
     * is guaranteed not to be exeeded in any invocation of the kernel. Exceeding
     * the the maximum number of threads results in runtime error or kernel launch
     * failure. For kernels already using PTX directive .maxntid, this option will
     * be ignored by default. Use ::CU_JIT_OVERRIDE_DIRECTIVE_VALUES to let this
     * option take precedence over the PTX directive.
     * Option type: int\n
     * Applies to: compiler only
     */
    CU_JIT_MAX_THREADS_PER_BLOCK,

    /**
     * This option lets the values specified using ::CU_JIT_MAX_REGISTERS,
     * ::CU_JIT_THREADS_PER_BLOCK, ::CU_JIT_MAX_THREADS_PER_BLOCK and
     * ::CU_JIT_MIN_CTA_PER_SM take precedence over any PTX directives.
     * (0: Disable, default; 1: Enable)
     * Option type: int\n
     * Applies to: compiler only
     */
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES,

    CU_JIT_NUM_OPTIONS;
}
