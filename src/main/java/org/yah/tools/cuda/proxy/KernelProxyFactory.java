package org.yah.tools.cuda.proxy;

import org.yah.tools.cuda.proxy.invocation.KernelFunction;
import org.yah.tools.cuda.proxy.invocation.KernelFunctionFactory;
import org.yah.tools.cuda.proxy.invocation.KernelsInvocationHandler;
import org.yah.tools.cuda.proxy.services.DefaultServiceFactory;
import org.yah.tools.cuda.proxy.services.ListTypeWriterRegistry;
import org.yah.tools.cuda.proxy.services.ServiceFactory;
import org.yah.tools.cuda.proxy.services.TypeWriterRegistry;
import org.yah.tools.cuda.support.library.CudaModuleSupport;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static org.yah.tools.cuda.api.driver.Driver.CUfunction;
import static org.yah.tools.cuda.api.driver.Driver.CUmodule;
import static org.yah.tools.cuda.api.nvrtc.NVRTC.nvrtcProgram;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class KernelProxyFactory {

    private final ServiceFactory serviceFactory;
    private final TypeWriterRegistry typeWriterRegistry;

    public KernelProxyFactory() {
        this(new DefaultServiceFactory(), ListTypeWriterRegistry.create(true));
    }

    public KernelProxyFactory(ServiceFactory serviceFactory, TypeWriterRegistry typeWriterRegistry) {
        this.serviceFactory = Objects.requireNonNull(serviceFactory, "serviceFactory is null");
        this.typeWriterRegistry = Objects.requireNonNull(typeWriterRegistry, "typeWriterRegistry is null");
    }

    @SuppressWarnings("unchecked")
    public <T> T createKernelProxy(nvrtcProgram program, Class<T> nativeInterface) {
        List<Method> kernelMethods = collectKernelMethods(nativeInterface);
        CUmodule module = CudaModuleSupport.createModule(program);
        List<KernelFunction> kernelFunctions = new ArrayList<>();
        KernelFunctionFactory functionFactory = new KernelFunctionFactory(serviceFactory, typeWriterRegistry, name -> getKernelFunction(module, name));
        for (Method method : kernelMethods) {
            KernelFunction kernelFunction = functionFactory.create(method);
            if (kernelFunction != null)
                kernelFunctions.add(kernelFunction);
        }
        InvocationHandler invocationHandler = new KernelsInvocationHandler(module, kernelFunctions);
        return (T) Proxy.newProxyInstance(nativeInterface.getClassLoader(), new Class<?>[]{nativeInterface}, invocationHandler);
    }

    private CUfunction getKernelFunction(CUmodule module, String name) {
        CUfunction.ByReference hfunc = new CUfunction.ByReference();
        check(driverAPI().cuModuleGetFunction(hfunc, module, name));
        return hfunc.getValue();
    }

    private static List<Method> collectKernelMethods(Class<?> container) {
        List<Method> methods = new ArrayList<>();
        collectKernelMethods(container, methods);
        return methods;
    }

    private static void collectKernelMethods(Class<?> container, List<Method> methods) {
        Class<?>[] interfaces = container.getInterfaces();
        for (Class<?> parent : interfaces) {
            collectKernelMethods(parent, methods);
        }
        Method[] kernelMethods = container.getDeclaredMethods();
        methods.addAll(Arrays.asList(kernelMethods));
    }

}
