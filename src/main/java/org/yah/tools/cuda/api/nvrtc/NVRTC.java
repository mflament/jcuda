package org.yah.tools.cuda.api.nvrtc;

import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.NativeEnum;
import org.yah.tools.cuda.support.NVRTCSupport;

import javax.annotation.Nullable;

public interface NVRTC extends Library {

    class nvrtcProgram extends Pointer implements AutoCloseable {
        public nvrtcProgram(long peer) {
            super(peer);
        }

        public nvrtcProgram(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        public Memory getPTX() {
            Memory memory = new Memory(getPTXSize());
            NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTX(this, memory));
            return memory;
        }

        public long getPTXSize() {
            PointerByReference sizeRet = new PointerByReference();
            NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcGetPTXSize(this, sizeRet));
            return Pointer.nativeValue(sizeRet.getValue());
        }

        @Override
        public void close() {
            NVRTCSupport.check(NVRTCSupport.nvrtc().nvrtcDestroyProgram(new ByReference(this)));
        }

        public static class ByReference extends PointerByReference {
            public ByReference() {
            }

            public ByReference(nvrtcProgram value) {
                super(value);
            }

            @Override
            public nvrtcProgram getValue() {
                return new nvrtcProgram(super.getValue());
            }
        }
    }

    /**
     * \ingroup error
     * \brief   The enumerated type nvrtcResult defines API call result codes.
     * NVRTC API functions return nvrtcResult to indicate the call
     * result.
     */
    enum nvrtcResult implements NativeEnum {
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

    int nvrtcVersion(Pointer major, Pointer minor);

    Pointer nvrtcGetErrorString(nvrtcResult result);

    nvrtcResult nvrtcCreateProgram(nvrtcProgram.ByReference prog, String src, @Nullable String name, int numHeaders, @Nullable String[] headers, @Nullable String[] includeNames);

    nvrtcResult nvrtcDestroyProgram(nvrtcProgram.ByReference prog);

    nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, Pointer options);

    nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, PointerByReference logSizeRet);

    nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, Pointer log);

    nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, PointerByReference ptxSizeRet);

    nvrtcResult nvrtcGetPTX(nvrtcProgram prog, Pointer ptx);

    nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, PointerByReference cubinSizeRet);

    nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, Pointer cubin);

}
