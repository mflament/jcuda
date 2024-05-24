package org.yah.tools.cuda.support.library;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.support.CudaException;

import java.nio.file.Path;

import static org.yah.tools.cuda.api.driver.Driver.CUmodule;
import static org.yah.tools.cuda.api.nvrtc.NVRTC.nvrtcProgram;
import static org.yah.tools.cuda.api.runtime.Runtime.CUDA_ERROR_INVALID_IMAGE;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;

public class CudaModuleSupport {

    private static final Logger LOGGER = LoggerFactory.getLogger(CudaModuleSupport.class);

    public static CUmodule createModule(nvrtcProgram program) {
        try (Memory ptx = program.getPTX()) {
            CUmodule.ByReference moduleRef = new CUmodule.ByReference();
            try {
                check(driverAPI().cuModuleLoadData(moduleRef, ptx));
            } catch (CudaException e) {
                if (e.getStatus() == CUDA_ERROR_INVALID_IMAGE)
                    LOGGER.info("PTX:\n{}", readNTS(ptx, ptx.size()));
                throw e;
            }
            return moduleRef.getValue();
        }
    }

    public static CUmodule createModule(Path file) {
        CUmodule.ByReference reference = new CUmodule.ByReference();
        check(driverAPI().cuModuleLoad(reference, file.toString()));
        return reference.getValue();
    }

    public static CUmodule createModule(Pointer image) {
        CUmodule.ByReference reference = new CUmodule.ByReference();
        check(driverAPI().cuModuleLoadData(reference, image));
        return reference.getValue();
    }

}
