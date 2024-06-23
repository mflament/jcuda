package org.yah.tools.cuda.api.driver;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class CUKernel extends Pointer {
    public CUKernel(long peer) {
        super(peer);
    }

    public CUKernel(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUKernel getValue() {
            return new CUKernel(super.getValue());
        }
    }
}
