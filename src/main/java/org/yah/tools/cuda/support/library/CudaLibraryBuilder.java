package org.yah.tools.cuda.support.library;

import com.sun.jna.Pointer;
import org.yah.tools.cuda.api.driver.CUjit_option;
import org.yah.tools.cuda.api.driver.CUlibraryOption;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;

import org.yah.tools.cuda.api.driver.CULibrary;
import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class CudaLibraryBuilder {
    private final Pointer image;
    private final Path file;

    private final Map<CUjit_option, Pointer> jitOptions = new EnumMap<>(CUjit_option.class);
    private final Map<CUlibraryOption, Pointer> libraryOptions = new EnumMap<>(CUlibraryOption.class);

    public CudaLibraryBuilder(Pointer image) {
        this.image = image;
        this.file = null;
    }

    public CudaLibraryBuilder(Path file) {
        this.file = file;
        this.image = null;
    }

    public Map<CUjit_option, Pointer> jitOptions() {
        return jitOptions;
    }

    public CudaLibraryBuilder jitOption(CUjit_option option, Pointer value) {
        this.jitOptions.put(option, value);
        return this;
    }

    public Map<CUlibraryOption, Pointer> libraryOptions() {
        return libraryOptions;
    }

    public CudaLibraryBuilder libraryOption(CUlibraryOption option, Pointer value) {
        this.libraryOptions.put(option, value);
        return this;
    }

    public CULibrary build() {
        if (image == null && file == null)
            throw new IllegalStateException("code or file is required");

        CULibrary.ByReference libraryRef = new CULibrary.ByReference();
        CUjit_option[] jitOptionsNames = null;
        CUlibraryOption[] libraryOptionsNames = null;
        Pointer[] jitOptionsValues = null, libraryOptionsValues = null;
        if (!jitOptions.isEmpty()) {
            jitOptionsNames = new CUjit_option[jitOptions.size()];
            jitOptionsValues = new Pointer[jitOptions.size()];
            @SuppressWarnings("unchecked") Map.Entry<CUjit_option, Pointer>[] entries = jitOptions.entrySet().toArray(Map.Entry[]::new);
            for (int i = 0; i < entries.length; i++) {
                jitOptionsNames[i] = entries[i].getKey();
                jitOptionsValues[i] = entries[i].getValue();
            }
        }

        if (!libraryOptions.isEmpty()) {
            libraryOptionsNames = new CUlibraryOption[libraryOptions.size()];
            libraryOptionsValues = new Pointer[libraryOptions.size()];
            @SuppressWarnings("unchecked") Map.Entry<CUlibraryOption, Pointer>[] entries = libraryOptions.entrySet().toArray(Map.Entry[]::new);
            for (int i = 0; i < entries.length; i++) {
                libraryOptionsNames[i] = entries[i].getKey();
                libraryOptionsValues[i] = entries[i].getValue();
            }
        }

        if (image != null) {
            cuCheck(driverAPI().cuLibraryLoadData(libraryRef, image, jitOptionsNames, jitOptionsValues, jitOptions.size(),
                    libraryOptionsNames, libraryOptionsValues, libraryOptions.size()));
        } else {
            cuCheck(driverAPI().cuLibraryLoadFromFile(libraryRef, file.toAbsolutePath().toString(), jitOptionsNames, jitOptionsValues, jitOptions.size(),
                    libraryOptionsNames, libraryOptionsValues, libraryOptions.size()));
        }
        return libraryRef.getValue();
    }

}
