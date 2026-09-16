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

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/backends/amdgpu_runtime.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

struct AmdgpuBackend {
  using StreamType = hipStream_t;
  using ContextType = hipCtx_t;
  using KernelHandle = hipFunction_t;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
    bool cooperative = false;
  };

  struct ModuleData {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    AmdgpuKernelMetadata metadata;
  };

  static inline std::unordered_map<std::string, ModuleData> module_cache_;
  static inline std::mutex cache_mutex_;

  static LaunchOptions prepare_launch(const std::string& dir,
                                      const std::string& kernel_name,
                                      unsigned int shared_mem,
                                      const std::string& /*sig*/,
                                      size_t /*num_args*/) {
    AmdgpuKernelMetadata metadata = get_loaded_metadata(dir, kernel_name);
    if (metadata.global_scratch_size != 0 || metadata.profile_scratch_size != 0) {
      throw std::runtime_error(fmt::format(
          "AMDGPU kernel {} requests unsupported scratch storage "
          "(global={}, profile={})",
          kernel_name,
          metadata.global_scratch_size,
          metadata.profile_scratch_size));
    }
    return {.shared_memory = shared_mem, .cooperative = metadata.launch_cooperative_grid};
  }

  static void launch_kernel(hipStream_t stream,
                            hipFunction_t kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
      return;
    }

    hipError_t result;
    if (opts.cooperative) {
      result = amdgpu::Runtime::module_launch_cooperative_kernel(kernel,
                                                                grid_x,
                                                                grid_y,
                                                                grid_z,
                                                                block_x,
                                                                block_y,
                                                                block_z,
                                                                opts.shared_memory,
                                                                stream,
                                                                args);
    } else {
      result = amdgpu::Runtime::module_launch_kernel(kernel,
                                                     grid_x,
                                                     grid_y,
                                                     grid_z,
                                                     block_x,
                                                     block_y,
                                                     block_z,
                                                     opts.shared_memory,
                                                     stream,
                                                     args,
                                                     nullptr);
    }

    if (result != hipSuccess) {
      throw std::runtime_error(fmt::format("AMDGPU kernel launch failed: {}",
                                           amdgpu::Runtime::get_error_string(result)));
    }
  }

  static void ensure_context() {
    int device_count = 0;
    checkAmdgpuErrors(amdgpu::Runtime::get_device_count(&device_count));
    if (device_count == 0) {
      throw std::runtime_error("No AMDGPU devices are visible to the HIP runtime");
    }

    int device_id = 0;
    if (amdgpu::Runtime::get_device(&device_id) != hipSuccess) {
      device_id = 0;
    }
    checkAmdgpuErrors(amdgpu::Runtime::set_device(device_id));
  }

  static int get_device_index() {
    int device_id = 0;
    checkAmdgpuErrors(amdgpu::Runtime::get_device(&device_id));
    return device_id;
  }

  static hipFunction_t load_kernel(const std::string& dir, const std::string& kernel_name) {
    const int device_id = get_device_index();
    const std::string key = make_cache_key(dir, kernel_name, device_id);

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto cached = module_cache_.find(key);
    if (cached != module_cache_.end()) {
      return cached->second.function;
    }

    AmdgpuKernelMetadata metadata = load_amdgpu_metadata(dir, kernel_name);
    if (metadata.arch.empty()) {
      throw std::runtime_error(
          fmt::format("AMDGPU metadata for kernel '{}' has no target architecture", kernel_name));
    }
    if (!metadata.target_backend.empty() && metadata.target_backend != "hip") {
      throw std::runtime_error(fmt::format(
          "AMDGPU kernel '{}' was compiled for backend '{}', expected 'hip'",
          kernel_name,
          metadata.target_backend));
    }
    if (metadata.warp_size != 32 && metadata.warp_size != 64) {
      throw std::runtime_error(fmt::format(
          "AMDGPU kernel '{}' has invalid or missing warp_size {} in its Triton metadata",
          kernel_name,
          metadata.warp_size));
    }
    if (metadata.num_ctas != 1) {
      throw std::runtime_error(fmt::format(
          "AMDGPU kernel '{}' requests num_ctas={}, but clustered launches are not supported",
          kernel_name,
          metadata.num_ctas));
    }

    hipDeviceProp_t properties;
    checkAmdgpuErrors(amdgpu::Runtime::get_device_properties(&properties, device_id));
    const std::string device_arch = base_architecture(properties.gcnArchName);
    const std::string kernel_arch = base_architecture(metadata.arch);
    if (device_arch != kernel_arch) {
      throw std::runtime_error(fmt::format(
          "AMDGPU architecture mismatch: device is {}, kernel requires {}",
          device_arch,
          kernel_arch));
    }

    if (metadata.shared > static_cast<size_t>(properties.sharedMemPerBlock)) {
      throw std::runtime_error(fmt::format(
          "AMDGPU kernel '{}' requests {} bytes of shared memory, but device {} supports {}",
          kernel_name,
          metadata.shared,
          device_id,
          properties.sharedMemPerBlock));
    }

    const std::string hsaco_path = fmt::format("{}/{}.hsaco", dir, kernel_name);
    hipModule_t module = nullptr;
    checkAmdgpuErrors(amdgpu::Runtime::module_load(&module, hsaco_path.c_str()));

    const std::string& symbol_name =
        metadata.symbol_name.empty() ? kernel_name : metadata.symbol_name;
    hipFunction_t function = nullptr;
    hipError_t function_result =
        amdgpu::Runtime::module_get_function(&function, module, symbol_name.c_str());
    if (function_result != hipSuccess) {
      (void)amdgpu::Runtime::module_unload(module);
      throw std::runtime_error(fmt::format("Failed to find AMDGPU kernel symbol '{}' in {}: {}",
                                           symbol_name,
                                           hsaco_path,
                                           amdgpu::Runtime::get_error_string(function_result)));
    }

    LOG(INFO) << fmt::format("Loaded AMDGPU kernel {} for {} (wave{}, shared={} bytes)",
                             kernel_name,
                             kernel_arch,
                             metadata.warp_size,
                             metadata.shared);
    module_cache_.emplace(key, ModuleData {module, function, std::move(metadata)});
    return function;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    // TritonKernelImpl caches this value before lazy-loading the module, so it
    // must be obtainable directly from the compiled kernel metadata.
    return load_amdgpu_metadata(dir, kernel_name).shared;
  }

  static unsigned int get_warp_size(const std::string& dir, const std::string& kernel_name) {
    return get_loaded_metadata(dir, kernel_name).warp_size;
  }

 private:
  static std::string base_architecture(std::string arch) {
    const size_t feature_separator = arch.find(':');
    if (feature_separator != std::string::npos) {
      arch.resize(feature_separator);
    }
    return arch;
  }

  static std::string make_cache_key(const std::string& dir,
                                    const std::string& kernel_name,
                                    int device_id) {
    return fmt::format("{}::{}::{}", device_id, dir, kernel_name);
  }

  static AmdgpuKernelMetadata get_loaded_metadata(const std::string& dir,
                                                  const std::string& kernel_name) {
    const int device_id = get_device_index();
    const std::string key = make_cache_key(dir, kernel_name, device_id);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto cached = module_cache_.find(key);
    if (cached == module_cache_.end()) {
      throw std::runtime_error(
          fmt::format("AMDGPU kernel '{}' has not been loaded", kernel_name));
    }
    return cached->second.metadata;
  }
};

static_assert(BackendPolicy<AmdgpuBackend>, "AmdgpuBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
