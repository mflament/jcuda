package org.yah.tools.cuda.api.driver;

import org.yah.tools.cuda.api.NativeEnum;

/**
 * Function cache configurations
 */
public enum CUfunc_cache implements NativeEnum {
    CU_FUNC_CACHE_PREFER_NONE,
    /**
     * < no preference for shared memory or L1 (default)
     */
    CU_FUNC_CACHE_PREFER_SHARED,
    /**
     * < prefer larger shared memory and smaller L1 cache
     */
    CU_FUNC_CACHE_PREFER_L1,
    /**
     * < prefer larger L1 cache and smaller shared memory
     */
    CU_FUNC_CACHE_PREFER_EQUAL;  // < prefer equal sized L1 cache and shared memory
}
