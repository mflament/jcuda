package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class CUcontext extends Pointer implements AutoCloseable {
    public CUcontext(long peer) {
        super(peer);
    }

    public CUcontext(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public void setCurrent() {
        cuCheck(driverAPI().cuCtxSetCurrent(this));
    }

    @Override
    public void close() {
        cuCheck(driverAPI().cuCtxDestroy(this));
    }

    public static CUcontext getCurrent() {
        ByReference ptrRef = new ByReference();
        cuCheck(driverAPI().cuCtxGetCurrent(ptrRef));
        return ptrRef.getValue();
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUcontext getValue() {
            return new CUcontext(super.getValue());
        }
    }
}
