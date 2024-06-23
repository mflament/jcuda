package org.yah.tools.cuda.api;

import com.sun.jna.FromNativeContext;
import com.sun.jna.NativeMapped;
import org.yah.tools.cuda.api.driver.CUctx_flags;

import java.util.Collection;
import java.util.EnumSet;
import java.util.Set;

public interface NativeEnum extends NativeMapped {

    static int all(Collection<CUctx_flags> flags) {
        int res = 0;
        for (CUctx_flags flag : flags) res |= flag.nativeValue();
        return res;
    }

    int ordinal();

    String name();

    default int nativeValue() {
        return ordinal();
    }

    @Override
    default Object fromNative(Object nativeValue, FromNativeContext context) {
        Class<?> targetType = context.getTargetType();
        int value = (Integer) nativeValue;
        NativeEnum[] enumConstants = (NativeEnum[]) targetType.getEnumConstants();
        for (NativeEnum enumConstant : enumConstants) {
            if (enumConstant.nativeValue() == value)
                return enumConstant;
        }
        throw new IllegalArgumentException("unresolved enum '" + targetType.getName() + "' constant for native value " + value);
    }

    @Override
    default Object toNative() {
        return nativeValue();
    }

    @Override
    default Class<?> nativeType() {
        return int.class;
    }
}
