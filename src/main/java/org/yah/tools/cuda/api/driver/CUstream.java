package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class CUstream extends Pointer {
    public CUstream(long peer) {
        super(peer);
    }

    public CUstream(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUstream getValue() {
            return new CUstream(super.getValue());
        }
    }
}
