package org.yah.tools.cuda.support.library;

import com.sun.jna.Pointer;
import org.yah.tools.cuda.api.driver.CUjit_option;
import org.yah.tools.cuda.api.driver.CUlibraryOption;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;

import static org.yah.tools.cuda.api.driver.Driver.CULibrary;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class CudaLibraryBuilder {
    private Pointer code;
    private Path file;
    private final Map<CUjit_option, Object> jitOptions = new EnumMap<>(CUjit_option.class);
    private final Map<CUlibraryOption, Object> libraryOptions = new EnumMap<>(CUlibraryOption.class);

    public Pointer code() {
        return code;
    }

    public CudaLibraryBuilder code(Pointer code) {
        this.code = code;
        return this;
    }

    public Path file() {
        return file;
    }

    public CudaLibraryBuilder file(Path file) {
        this.file = file;
        return this;
    }

    public Map<CUjit_option, Object> jitOptions() {
        return jitOptions;
    }

    public CudaLibraryBuilder jitOption(CUjit_option option, Object value) {
        this.jitOptions.put(option, value);
        return this;
    }

    public Map<CUlibraryOption, Object> libraryOptions() {
        return libraryOptions;
    }

    public CudaLibraryBuilder libraryOption(CUlibraryOption option, Object value) {
        this.libraryOptions.put(option, value);
        return this;
    }

    public CULibrary build() {
        if (code == null && file == null)
            throw new IllegalStateException("code or file is required");

        CULibrary.ByReference libraryRef = new CULibrary.ByReference();
        CUjit_option[] jitOptionsNames = null;
        CUlibraryOption[] libraryOptionsNames = null;
        Pointer[] jitOptionsValues = null, libraryOptionsValues = null;
        if (!jitOptions.isEmpty()) {
            jitOptionsNames = getJitOptionNames();
            throw new UnsupportedOperationException("TODO : handle jit options values");
        }
        if (!libraryOptions.isEmpty()) {
            libraryOptionsNames = getLibraryOptionNames();
            throw new UnsupportedOperationException("TODO : handle library options values");
        }
        if (code != null) {
            check(driverAPI().cuLibraryLoadData(libraryRef, code, jitOptionsNames, jitOptionsValues, jitOptions.size(),
                    libraryOptionsNames, libraryOptionsValues, libraryOptions.size()));
        } else {
            check(driverAPI().cuLibraryLoadFromFile(libraryRef, file.toAbsolutePath().toString(), jitOptionsNames, jitOptionsValues, jitOptions.size(),
                    libraryOptionsNames, libraryOptionsValues, libraryOptions.size()));
        }
        return libraryRef.getValue();
    }

    private CUjit_option[] getJitOptionNames() {
        if (jitOptions.isEmpty())
            return null;
        return jitOptions.keySet().toArray(CUjit_option[]::new);
    }

    private CUlibraryOption[] getLibraryOptionNames() {
        if (libraryOptions.isEmpty())
            return null;
        return libraryOptions.keySet().toArray(CUlibraryOption[]::new);
    }

    private Pointer allocateValues(Map<? extends Enum<?>, Object> options) {
        if (options.isEmpty())
            return null;
        throw new UnsupportedOperationException("TODO");
    }

}
