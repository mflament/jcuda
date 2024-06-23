package org.yah.tools.cuda.api.runtime;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.driver.CUfunction;

public class cudaStream extends Pointer {

    public cudaStream(long peer) {
        super(peer);
    }

    public cudaStream(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public cudaStream getValue() {
            return new cudaStream(super.getValue());
        }
    }

}
