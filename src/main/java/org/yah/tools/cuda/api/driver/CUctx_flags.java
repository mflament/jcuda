package org.yah.tools.cuda.api.driver;

import org.yah.tools.cuda.api.NativeEnum;

/**
 * Context creation flags
 */
public enum CUctx_flags implements NativeEnum {
    /**
     * < Automatic scheduling
     */
    CU_CTX_SCHED_AUTO(0x00),
    /**
     * < Set spin as default scheduling
     */
    CU_CTX_SCHED_SPIN(0x01),
    /**
     * < Set yield as default scheduling
     */
    CU_CTX_SCHED_YIELD(0x02),
    /**
     * < Set blocking synchronization as default scheduling
     */
    CU_CTX_SCHED_BLOCKING_SYNC(0x04),
    /**
     * < Set blocking synchronization as default scheduling
     *
     * @deprecated This flag was deprecated as of CUDA 4.0
     * and was replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
     */
    @Deprecated
    CU_CTX_BLOCKING_SYNC(0x04),
    CU_CTX_SCHED_MASK(0x07),
    /**
     * @deprecated This flag was deprecated as of CUDA 11.0 and it no longer has any effect.
     * All contexts as of CUDA 3.2 behave as though the flag is enabled.
     */
    @Deprecated
    CU_CTX_MAP_HOST(0x08),
    /**
     * < Keep local memory allocation after launch
     */
    CU_CTX_LMEM_RESIZE_TO_MAX(0x10),
    /**
     * < Trigger coredumps from exceptions in this context
     */
    CU_CTX_COREDUMP_ENABLE(0x20),
    /**
     * < Enable user pipe to trigger coredumps in this context
     */
    CU_CTX_USER_COREDUMP_ENABLE(0x40),
    /**
     * < Ensure synchronous memory operations on this context will synchronize
     */
    CU_CTX_SYNC_MEMOPS(0x80),
    CU_CTX_FLAGS_MASK(0xFF);

    private final int nativeValue;

    CUctx_flags(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    @Override
    public int nativeValue() {
        return nativeValue;
    }
}
