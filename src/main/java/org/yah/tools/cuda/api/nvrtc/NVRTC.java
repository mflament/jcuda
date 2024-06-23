package org.yah.tools.cuda.api.nvrtc;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import javax.annotation.Nullable;

public interface NVRTC extends Library {

    int nvrtcVersion(Pointer major, Pointer minor);

    Pointer nvrtcGetErrorString(nvrtcResult result);

    nvrtcResult nvrtcCreateProgram(PointerByReference prog, String src, @Nullable String name, int numHeaders, @Nullable String[] headers, @Nullable String[] includeNames);

    nvrtcResult nvrtcDestroyProgram(nvrtcProgram.ByReference prog);

    nvrtcResult nvrtcCompileProgram(Pointer prog, int numOptions, Pointer options);

    nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, Pointer options);

    nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, PointerByReference logSizeRet);

    nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, Pointer log);

    nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, PointerByReference ptxSizeRet);

    nvrtcResult nvrtcGetPTX(nvrtcProgram prog, Pointer ptx);

    nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, PointerByReference cubinSizeRet);

    nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, Pointer cubin);

    nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, String name_expression);

    nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, String name_expression, PointerByReference lowered_name);

}
