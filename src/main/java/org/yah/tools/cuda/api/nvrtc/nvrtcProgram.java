package org.yah.tools.cuda.api.nvrtc;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.support.NVRTCSupport;
import org.yah.tools.cuda.support.NativeSupport;

import javax.annotation.Nullable;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.yah.tools.cuda.support.NVRTCSupport.nvrtc;
import static org.yah.tools.cuda.support.NVRTCSupport.nvrtcCheck;

public class nvrtcProgram extends Pointer implements AutoCloseable {
    private final Map<String, String> loweredNames = new HashMap<>();

    public nvrtcProgram(long peer) {
        super(peer);
    }

    public nvrtcProgram(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public Memory getPTX() {
        Memory memory = new Memory(getPTXSize());
        nvrtcCheck(NVRTCSupport.nvrtc().nvrtcGetPTX(this, memory));
        return memory;
    }

    public long getPTXSize() {
        PointerByReference sizeRet = new PointerByReference();
        nvrtcCheck(NVRTCSupport.nvrtc().nvrtcGetPTXSize(this, sizeRet));
        return Pointer.nativeValue(sizeRet.getValue());
    }

    public Memory getCUBIN() {
        Memory memory = new Memory(getCUBINSize());
        nvrtcCheck(NVRTCSupport.nvrtc().nvrtcGetCUBIN(this, memory));
        return memory;
    }

    public long getCUBINSize() {
        PointerByReference sizeRet = new PointerByReference();
        nvrtcCheck(NVRTCSupport.nvrtc().nvrtcGetCUBINSize(this, sizeRet));
        return Pointer.nativeValue(sizeRet.getValue());
    }

    @Nullable
    public String getLoweredName(String nameExpression) {
        return loweredNames.computeIfAbsent(nameExpression, this::resolveLoweredName);
    }

    private String resolveLoweredName(String nameExpression) {
        PointerByReference nameRef = new PointerByReference();
        nvrtcCheck(nvrtc().nvrtcGetLoweredName(this, nameExpression, nameRef));
        return NativeSupport.readNTS(nameRef.getValue());
    }

    @Override
    public void close() {
        nvrtcCheck(NVRTCSupport.nvrtc().nvrtcDestroyProgram(new ByReference(this)));
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
