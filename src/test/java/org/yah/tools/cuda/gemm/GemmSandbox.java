package org.yah.tools.cuda.gemm;

import com.sun.jna.Memory;
import org.yah.tools.cuda.TestsHelper;
import org.yah.tools.cuda.api.driver.CUcontext;
import org.yah.tools.cuda.api.driver.CUdevice;
import org.yah.tools.cuda.api.driver.Driver;
import org.yah.tools.cuda.api.nvrtc.nvrtcProgram;
import org.yah.tools.cuda.api.runtime.Runtime;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.NativeSupport;
import org.yah.tools.cuda.support.RuntimeSupport;
import org.yah.tools.cuda.support.program.NVRTCProgramBuilder;

public class GemmSandbox {

    public static void main(String[] args) {
        Runtime runtime = RuntimeSupport.runtimeAPI();
        Driver driver = DriverSupport.driverAPI();
        runtime.cudaSetDevice(0);

        String src = TestsHelper.loadSource("gemm.cu");
        CUdevice device = DriverSupport.getDevice(0);
        try (CUcontext ctx = device.createContext()) {
            ctx.setCurrent();

            nvrtcProgram program = NVRTCProgramBuilder.create(src)
                    .programName("gemm")
                    .smVersion(device)
                    .build();

            Memory ptx = program.getPTX();
            System.out.println(NativeSupport.readNTS(ptx, ptx.size()));
        }
    }
}
