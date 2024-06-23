package org.yah.tools.cuda.executor;

public class dim3 {
    public int x;
    public int y;
    public int z;

    public dim3() {
    }

    public dim3(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public dim3(dim3 from) {
        this(from.x, from.y, from.z);
    }

    public int x() {
        return x;
    }

    public int y() {
        return y;
    }

    public int z() {
        return z;
    }

    public dim3 x(int x) {
        this.x = x;
        return this;
    }

    public dim3 y(int y) {
        this.y = y;
        return this;
    }

    public dim3 z(int z) {
        this.z = z;
        return this;
    }

    @Override
    public String toString() {
        return "dim3{" +
                "x=" + x +
                ", y=" + y +
                ", z=" + z +
                '}';
    }
}
