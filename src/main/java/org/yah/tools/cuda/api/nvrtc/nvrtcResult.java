package org.yah.tools.cuda.api.nvrtc;

import org.yah.tools.cuda.api.NativeEnum;

/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 * NVRTC API functions return nvrtcResult to indicate the call
 * result.
 */
public enum nvrtcResult implements NativeEnum {
    NVRTC_SUCCESS,
    NVRTC_ERROR_OUT_OF_MEMORY,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
    NVRTC_ERROR_INVALID_INPUT,
    NVRTC_ERROR_INVALID_PROGRAM,
    NVRTC_ERROR_INVALID_OPTION,
    NVRTC_ERROR_COMPILATION,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
    NVRTC_ERROR_INTERNAL_ERROR,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED
}
