package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class CUfunction extends Pointer {
    public CUfunction(long peer) {
        super(peer);
    }

    public CUfunction(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUfunction getValue() {
            return new CUfunction(super.getValue());
        }
    }
}
