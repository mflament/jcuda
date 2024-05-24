package org.yah.tools.cuda.api.driver;

import org.yah.tools.cuda.api.NativeEnum;

public enum CUlibraryOption implements NativeEnum {
    CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE,

    /**
     * Specifes that the argument \p code passed to ::cuLibraryLoadData() will be preserved.
     * Specifying this option will let the driver know that \p code can be accessed at any point
     * until ::cuLibraryUnload(). The default behavior is for the driver to allocate and
     * maintain its own copy of \p code. Note that this is only a memory usage optimization
     * hint and the driver can choose to ignore it if required.
     * Specifying this option with ::cuLibraryLoadFromFile() is invalid and
     * will return ::CUDA_ERROR_INVALID_VALUE.
     */
    CU_LIBRARY_BINARY_IS_PRESERVED,

    CU_LIBRARY_NUM_OPTIONS;
}
