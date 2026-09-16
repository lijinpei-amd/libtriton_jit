// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <stdexcept>
#include <string>

#define TRITON_JIT_AMDGPU_STRINGIFY_IMPL(name) #name
#define TRITON_JIT_AMDGPU_STRINGIFY(name) TRITON_JIT_AMDGPU_STRINGIFY_IMPL(name)

namespace triton_jit::amdgpu {

// PyTorch wheels may bundle a HIP runtime whose library name differs from the
// system ROCm SONAME. Linking another HIP runtime into the same process can
// create two independent sets of process-global HIP state. Resolve the C API
// from the runtime already loaded by PyTorch instead.
class Runtime {
 public:
  static const char* get_error_string(hipError_t error) {
    return function<decltype(&::hipGetErrorString)>(TRITON_JIT_AMDGPU_STRINGIFY(hipGetErrorString))(
        error);
  }

  static hipError_t get_device_count(int* count) {
    return function<decltype(&::hipGetDeviceCount)>(TRITON_JIT_AMDGPU_STRINGIFY(hipGetDeviceCount))(
        count);
  }

  static hipError_t get_device(int* device) {
    return function<decltype(&::hipGetDevice)>(TRITON_JIT_AMDGPU_STRINGIFY(hipGetDevice))(device);
  }

  static hipError_t set_device(int device) {
    return function<decltype(&::hipSetDevice)>(TRITON_JIT_AMDGPU_STRINGIFY(hipSetDevice))(device);
  }

  static hipError_t get_device_properties(hipDeviceProp_t* properties, int device) {
    return function<decltype(&::hipGetDeviceProperties)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipGetDeviceProperties))(properties, device);
  }

  static hipError_t module_load(hipModule_t* module, const char* path) {
    return function<decltype(&::hipModuleLoad)>(TRITON_JIT_AMDGPU_STRINGIFY(hipModuleLoad))(
        module, path);
  }

  static hipError_t module_unload(hipModule_t module) {
    return function<decltype(&::hipModuleUnload)>(TRITON_JIT_AMDGPU_STRINGIFY(hipModuleUnload))(
        module);
  }

  static hipError_t module_get_function(hipFunction_t* function_handle,
                                        hipModule_t module,
                                        const char* symbol_name) {
    return function<decltype(&::hipModuleGetFunction)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipModuleGetFunction))(function_handle, module, symbol_name);
  }

  static hipError_t module_launch_kernel(hipFunction_t function_handle,
                                         unsigned int grid_x,
                                         unsigned int grid_y,
                                         unsigned int grid_z,
                                         unsigned int block_x,
                                         unsigned int block_y,
                                         unsigned int block_z,
                                         unsigned int shared_memory,
                                         hipStream_t stream,
                                         void** kernel_params,
                                         void** extra) {
    return function<decltype(&::hipModuleLaunchKernel)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipModuleLaunchKernel))(
        function_handle,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_memory,
        stream,
        kernel_params,
        extra);
  }

  static hipError_t module_launch_cooperative_kernel(hipFunction_t function_handle,
                                                     unsigned int grid_x,
                                                     unsigned int grid_y,
                                                     unsigned int grid_z,
                                                     unsigned int block_x,
                                                     unsigned int block_y,
                                                     unsigned int block_z,
                                                     unsigned int shared_memory,
                                                     hipStream_t stream,
                                                     void** kernel_params) {
    return function<decltype(&::hipModuleLaunchCooperativeKernel)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipModuleLaunchCooperativeKernel))(function_handle,
                                                                      grid_x,
                                                                      grid_y,
                                                                      grid_z,
                                                                      block_x,
                                                                      block_y,
                                                                      block_z,
                                                                      shared_memory,
                                                                      stream,
                                                                      kernel_params);
  }

  static hipError_t device_synchronize() {
    return function<decltype(&::hipDeviceSynchronize)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipDeviceSynchronize))();
  }

  static hipError_t device_reset() {
    return function<decltype(&::hipDeviceReset)>(
        TRITON_JIT_AMDGPU_STRINGIFY(hipDeviceReset))();
  }

 private:
  template <typename Function>
  static Function function(const char* name) {
    dlerror();
    void* symbol = dlsym(RTLD_DEFAULT, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
      throw std::runtime_error("Unable to resolve HIP runtime symbol '" + std::string(name) +
                               "' from the runtime loaded by PyTorch: " +
                               (error != nullptr ? error : "symbol not found"));
    }
    return reinterpret_cast<Function>(symbol);
  }
};

}  // namespace triton_jit::amdgpu

#undef TRITON_JIT_AMDGPU_STRINGIFY
#undef TRITON_JIT_AMDGPU_STRINGIFY_IMPL
