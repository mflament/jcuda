package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class CULibrary extends Pointer implements AutoCloseable {
    public CULibrary(long peer) {
        super(peer);
    }

    public CULibrary(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    @Override
    public void close() {
        cuCheck(driverAPI().cuLibraryUnload(this));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CULibrary getValue() {
            return new CULibrary(super.getValue());
        }
    }
}
