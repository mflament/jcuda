package org.yah.tools.cuda.support.program;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.api.driver.CUdevice;
import org.yah.tools.cuda.api.nvrtc.NVRTC;
import org.yah.tools.cuda.api.nvrtc.nvrtcProgram;
import org.yah.tools.cuda.api.nvrtc.nvrtcResult;
import org.yah.tools.cuda.support.NVRTCSupport;
import org.yah.tools.cuda.support.NativeSupport;

import javax.annotation.Nullable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

import static org.yah.tools.cuda.support.NVRTCSupport.nvrtc;
import static org.yah.tools.cuda.support.NVRTCSupport.nvrtcCheck;

public class NVRTCProgramBuilder {

    private static final Logger LOGGER = LoggerFactory.getLogger(NVRTCProgramBuilder.class);

    public static NVRTCProgramBuilder create(String source) {
        Objects.requireNonNull(source, "source is null");
        return new NVRTCProgramBuilder(source);
    }

    private final String source;

    @Nullable
    private String programName;

    private String computeVersion;
    private String smVersion;

    private final List<Path> includeDirectories = new ArrayList<>();
    private final Map<String, String> includeEntries = new LinkedHashMap<>();
    private final List<String> compileOptions = new ArrayList<>();
    private final Set<String> nameExpressions = new LinkedHashSet<>();
    private boolean scanIncludes = true;

    private NVRTCProgramBuilder(String source) {
        this.source = source;
    }

    public String source() {
        return source;
    }

    @Nullable
    public String programName() {
        return programName;
    }

    public NVRTCProgramBuilder programName(@Nullable String programName) {
        this.programName = programName;
        return this;
    }

    public List<Path> includeDirectories() {
        return includeDirectories;
    }

    public NVRTCProgramBuilder includeDirectories(String... includeDirectories) {
        return includeDirectories(Arrays.stream(includeDirectories).map(Paths::get).toArray(Path[]::new));
    }

    public NVRTCProgramBuilder includeDirectories(Path... includeDirectories) {
        this.includeDirectories.addAll(Arrays.asList(includeDirectories));
        return this;
    }

    public NVRTCProgramBuilder nameExpression(String... nameExpressions) {
        this.nameExpressions.addAll(Arrays.asList(nameExpressions));
        return this;
    }

    public boolean scanIncludes() {
        return scanIncludes;
    }

    public NVRTCProgramBuilder scanIncludes(boolean scanIncludes) {
        this.scanIncludes = scanIncludes;
        return this;
    }

    public Map<String, String> includeEntries() {
        return includeEntries;
    }

    public NVRTCProgramBuilder include(String name) {
        return include(name, null);
    }

    public NVRTCProgramBuilder include(String name, @Nullable String content) {
        this.includeEntries.put(name, content);
        return this;
    }

    public List<String> compileOptions() {
        return compileOptions;
    }

    /**
     * @param options compile options from <a href="https://docs.nvidia.com/cuda/nvrtc/index.html#group__options">...</a>
     */
    public NVRTCProgramBuilder compileOptions(String... options) {
        this.compileOptions.addAll(Arrays.asList(options));
        return this;
    }

    /**
     * Helper to add compile options --gpu-architecture=<arch> (-arch)
     *
     * @param version version : (ie: 50,52, ... 90, 90a)
     * @see <a href="https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/">Matching CUDA arch and CUDA gencode for various NVIDIA architectures</a>
     */
    public NVRTCProgramBuilder computeVersion(String version) {
        this.computeVersion = version;
        return this;
    }

    public NVRTCProgramBuilder smVersion(String version) {
        this.smVersion = version;
        return this;
    }

    public NVRTCProgramBuilder computeVersion(CUdevice device) {
        return computeVersion(device.getComputeCapabilityVersion());
    }

    public NVRTCProgramBuilder smVersion(CUdevice device) {
        return smVersion(device.getComputeCapabilityVersion());
    }

    public nvrtcProgram build() {
        if (source == null)
            throw new IllegalStateException("No source configured");

        LOGGER.debug("building program {}", this);
        if (scanIncludes) {
            includeDirectories.forEach(this::scanIncludeDirectory);
        }
        Map<String, String> loadedIncludes = loadIncludes();
        String[] headers = null, includeNames = null;
        if (!loadedIncludes.isEmpty()) {
            headers = loadedIncludes.values().toArray(String[]::new);
            includeNames = loadedIncludes.keySet().toArray(String[]::new);
        }

        nvrtcProgram.ByReference ptrRef = new nvrtcProgram.ByReference();
        NVRTC nvrtc = nvrtc();
        nvrtcCheck(nvrtc.nvrtcCreateProgram(ptrRef, source, programName, loadedIncludes.size(), headers, includeNames));
        nvrtcProgram prog = new nvrtcProgram(ptrRef.getValue());
        Memory optionPtrs = null, optionsBuffer = null;

        if (smVersion != null)
            compileOptions(String.format("--gpu-architecture=sm_%s", smVersion));
        if (computeVersion != null)
            compileOptions(String.format("--gpu-architecture=compute_%s", computeVersion));
        if (!compileOptions.isEmpty()) {
            optionPtrs = new Memory((long) compileOptions.size() * Native.POINTER_SIZE);
            long optionsBufferSize = compileOptions.stream().mapToLong(s -> s.length() + 1).sum();
            optionsBuffer = new Memory(optionsBufferSize);
            Pointer optionPtr = optionsBuffer;
            for (int i = 0; i < compileOptions.size(); i++) {
                String compileOption = compileOptions.get(i);
                optionPtrs.setPointer(i * (long) Native.POINTER_SIZE, optionPtr);
                optionPtr = NativeSupport.writeNTS(optionPtr, compileOption);
            }
        }

        for (String nameExpression : nameExpressions) {
            nvrtcCheck(nvrtc.nvrtcAddNameExpression(prog, nameExpression));
        }

        nvrtcResult result = nvrtc.nvrtcCompileProgram(prog, compileOptions.size(), optionPtrs);
        if (result == nvrtcResult.NVRTC_ERROR_COMPILATION) {
            String programLog = NVRTCSupport.getProgramLog(prog);
            prog.close();
            throw new BuildProgramException(programLog);
        } else {
            NVRTCSupport.nvrtcCheck(result);
        }
        return prog;
    }

    private Map<String, String> loadIncludes() {
        Map<String, String> loadedEntries = new LinkedHashMap<>(includeEntries.size());
        includeEntries.forEach((name, content) -> {
            if (content == null) {
                content = loadIncludeFile(lookupIncludeFile(name));
            }
            loadedEntries.put(name, content);
        });
        return loadedEntries;
    }

    private Path lookupIncludeFile(String name) {
        for (Path includeDirectory : includeDirectories) {
            Path resolved = includeDirectory.resolve(name);
            if (Files.exists(resolved)) {
                return resolved;
            }
        }
        throw new RuntimeException("Include file for '" + name + "' not found in include directories " + includeDirectories);
    }

    private void scanIncludeDirectory(Path directory) {
        scanIncludeDirectory(directory, directory);
    }

    private void scanIncludeDirectory(Path directory, Path includeDir) {
        try (Stream<Path> entries = Files.list(directory)) {
            entries.forEach(child -> scanInclude(child, includeDir));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void scanInclude(Path path, Path includeDir) {
        if (Files.isDirectory(path))
            scanIncludeDirectory(path, includeDir);
        else {
            String content = loadIncludeFile(path);
            includeEntries.putIfAbsent(includeDir.relativize(path).toString(), content);
        }
    }

    private String loadIncludeFile(Path path) {
        try {
            return Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("Error loading include file '" + path + "'", e);
        }
    }

    @Override
    public String toString() {
        return "CudaProgramBuilder{" +
                "source='" + StringUtils.abbreviate(source, 25) + '\'' +
                ", programName='" + programName + '\'' +
                ", includeDirectories=" + includeDirectories +
                ", includeEntries=" + includeEntries +
                ", compileOptions=" + compileOptions +
                '}';
    }
}
