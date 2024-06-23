package org.yah.tools.cuda.support;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Null terminated string helper
 */
public final class NativeSupport {

    public static String readNTS(byte[] bytes) {
        int i;
        for (i = 0; i < bytes.length; i++) {
            if (bytes[i] == 0) break;
        }
        return new String(bytes, 0, i, StandardCharsets.US_ASCII);
    }

    public static String readNTS(Pointer pointer, long maxLength) {
        ByteBuffer byteBuffer = pointer.getByteBuffer(0, maxLength);
        StringBuilder sb = new StringBuilder();
        while (true) {
            byte b = byteBuffer.get();
            if (b == 0)
                break;
            sb.append((char) b);
        }
        return sb.toString();
    }

    public static String readNTS(Pointer pointer) {
        StringBuilder sb = new StringBuilder();
        int pos = 0;
        while (true) {
            byte b = pointer.getByte(pos++);
            if (b == 0)
                break;
            sb.append((char) b);
        }
        return sb.toString();
    }

    public static Pointer writeNTS(Pointer dst, String src) {
        byte[] bytes = src.getBytes(StandardCharsets.US_ASCII);
        dst.write(0, bytes,0, bytes.length);
        dst.setByte(bytes.length, (byte) 0);
        return dst.share(bytes.length + 1);
    }

    public static Memory loadText(Path file) throws IOException {
        return loadText(file, Charset.defaultCharset());
    }

    public static Memory loadText(Path file, Charset srcCharset) throws IOException {
        byte[] content = Files.readAllBytes(file);
        String str = new String(content, srcCharset);
        Memory pointer = new Memory(str.length() + 1);
        writeNTS(pointer, str);
        return pointer;
    }

    public static Memory loadFile(Path file) throws IOException {
        long size = Files.size(file);
        if (size > Integer.MAX_VALUE - 2)
            throw new UnsupportedOperationException("cubin size " + size + " exceed max size " + Integer.MAX_VALUE);
        Memory pointer = new Memory(size);
        byte[] content = Files.readAllBytes(file);
        pointer.write(0, content, 0, (int) size);
        return pointer;
    }

    private NativeSupport() {
    }
}
