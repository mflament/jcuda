package org.yah.tools.cuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.Test;
import org.yah.tools.cuda.api.driver.Driver;
import org.yah.tools.cuda.api.nvrtc.NVRTC;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.NativeSupport;

import static org.junit.jupiter.api.Assertions.*;
import static org.yah.tools.cuda.TestsHelper.loadSource;
import static org.yah.tools.cuda.api.driver.Driver.*;
import static org.yah.tools.cuda.api.nvrtc.NVRTC.*;

class NVRTCProgramBuilderTest {

    @Test
    void createAndCompileSucceed() {
        String src = loadSource("nvrtcTest.cu");
        CUdevice device = DriverSupport.getDevice(0);
        try (nvrtcProgram program = NVRTCProgramBuilder.create(src).smVersion(device).build()) {
            assertNotNull(program);
            assertNotEquals(0, Pointer.nativeValue(program));
            try (Memory ptx = program.getPTX()) {
                assertNotNull(ptx);
                assertEquals(program.getPTXSize(), ptx.size());
                System.out.println(NativeSupport.readNTS(ptx, ptx.size()));
            }
        }
    }

    @Test
    void createAndCompileWithError() {
        String src = loadSource("nvrtcTest_error.cu");
        CUdevice device = DriverSupport.getDevice(0);
        try (nvrtcProgram program = NVRTCProgramBuilder.create(src).computeVersion(device).programName("test_program").build()) {
            fail("Should have thrown BuildProgramException");
        } catch (BuildProgramException e) {
            assertTrue(e.getLog().contains("1 error detected in the compilation of \"test_program\"."));
        }
    }

}