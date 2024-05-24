package org.yah.tools.cuda.api;

import com.sun.jna.IntegerType;
import com.sun.jna.Native;

public class size_t extends IntegerType {
    private static final long serialVersionUID = 1L;

    public size_t() {
        this(0);
    }

    public size_t(long value) {
        super(Native.SIZE_T_SIZE, value);
    }
}
