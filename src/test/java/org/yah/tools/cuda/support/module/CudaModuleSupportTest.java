package org.yah.tools.cuda.support.module;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.junit.jupiter.api.Test;
import org.yah.tools.cuda.api.driver.CUcontext;
import org.yah.tools.cuda.api.driver.CUdevice;
import org.yah.tools.cuda.api.driver.CUfunction;
import org.yah.tools.cuda.api.driver.CUmodule;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.NativeSupport;
import org.yah.tools.cuda.support.library.CudaModuleSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.yah.tools.cuda.api.nvrtc.nvrtcProgram;
import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

class CudaModuleSupportTest {

    @Test
    void sandbox() throws IOException {
        CUdevice device = DriverSupport.getDevice(0);
        try (CUcontext ctx = device.createContext()) {
            ctx.setCurrent();

            Path file = Path.of("src/test/resources/nvrtcTest.ptx");
            Memory data = NativeSupport.loadFile(file);
            try (CUmodule module = CudaModuleSupport.createModule(data)) {
                CUfunction.ByReference hfunc = new CUfunction.ByReference();
                cuCheck(driverAPI().cuModuleGetFunction(hfunc, module, "_ZN7sandbox10TestKernel3addEiPdS1_S1_NS0_16KernelTestStructE"));
                Pointer funcPtr = hfunc.getValue();
                System.out.println(funcPtr);
            }
        }

    }

    private static ParsedPTX parsePTX(nvrtcProgram program) {
        try (Memory ptxBuffer = program.getPTX()) {
            String ptx = NativeSupport.readNTS(ptxBuffer, ptxBuffer.size());
            System.out.println(ptx);
            return parse(ptx);
        }
    }

    private static final Pattern CONST_PATTERN = Pattern.compile("\\.const \\..+? \\..+? (\\w+).*;");
    private static final Pattern ENTRY_PATTERN = Pattern.compile("\\.visible \\.entry (\\w+).*");

    private static ParsedPTX parse(String ptx) {
        List<String> constants = new ArrayList<>();
        List<String> entries = new ArrayList<>();
        String[] lines = ptx.split("\r?\n");
        for (String line : lines) {
            Matcher matcher = CONST_PATTERN.matcher(line);
            if (matcher.matches())
                constants.add(matcher.group(1));
            matcher = ENTRY_PATTERN.matcher(line);
            if (matcher.matches())
                entries.add(matcher.group(1));
        }
        return new ParsedPTX(constants, entries);
    }

    private record ParsedPTX(List<String> constants, List<String> entries) {
    }

}