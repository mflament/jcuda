package org.yah.tools.cuda.support;

import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.runtime.Runtime;
import org.yah.tools.cuda.api.runtime.cudaError;

import javax.annotation.Nullable;

public class RuntimeSupport {

    @Nullable
    private static Runtime runtime;

    public static synchronized Runtime runtimeAPI() {
        if (runtime == null)
            runtime = CudaSupport.createRuntimeApi();
        return runtime;
    }

    public static void cudaCheck(cudaError error) {
        if (error != cudaError.cudaSuccess)
            throw new CudaException("cudart", error);
    }

    public static String getRuntimeGetVersion() {
        PointerByReference ptr = new PointerByReference();
        try (Memory memory = new Memory(Integer.BYTES)) {
            cudaCheck(runtimeAPI().cudaRuntimeGetVersion(memory));
            int ver = memory.getInt(0);
            return String.format("%d.%d", ver / 1000, ver % 1000);
        }
    }
}
