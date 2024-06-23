package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class CUmodule extends Pointer implements AutoCloseable {
    public CUmodule(long peer) {
        super(peer);
    }

    public CUmodule(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    @Override
    public void close() {
        cuCheck(driverAPI().cuModuleUnload(this));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUmodule getValue() {
            return new CUmodule(super.getValue());
        }
    }
}
