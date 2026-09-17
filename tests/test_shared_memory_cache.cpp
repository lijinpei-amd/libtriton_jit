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

#include "triton_jit/triton_kernel.h"

#include <atomic>
#include <barrier>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct FakeBackend {
  using StreamType = void*;
  using ContextType = void*;
  using KernelHandle = int;

  static constexpr unsigned int WARP_SIZE = 32;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  inline static int shared_memory_query_count = 0;
  inline static int load_count = 0;
  inline static int prepare_launch_count = 0;
  inline static int launch_count = 0;
  inline static unsigned int shared_memory_value = 128;
  inline static unsigned int prepared_shared_memory = 0;
  inline static unsigned int launched_shared_memory = 0;

  static void ensure_context() {
  }

  static int get_device_index() {
    return 0;
  }

  static KernelHandle load_kernel(const std::string&, const std::string&) {
    ++load_count;
    return 1;
  }

  static unsigned int get_shared_memory(const std::string&, const std::string&) {
    ++shared_memory_query_count;
    return shared_memory_value;
  }

  static LaunchOptions prepare_launch(const std::string&,
                                      const std::string&,
                                      unsigned int shared_memory,
                                      const std::string&,
                                      size_t) {
    ++prepare_launch_count;
    prepared_shared_memory = shared_memory;
    return {.shared_memory = shared_memory};
  }

  static void launch_kernel(StreamType,
                            KernelHandle,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            void**,
                            const LaunchOptions& opts) {
    ++launch_count;
    launched_shared_memory = opts.shared_memory;
  }
};

static_assert(triton_jit::BackendPolicy<FakeBackend>);
static_assert(!triton_jit::KernelInvariantLaunchConfigBackend<FakeBackend>);

struct CachedLaunchBackend {
  using StreamType = void*;
  using ContextType = void*;
  using KernelHandle = int;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
    bool cooperative = false;
  };

  inline static std::atomic<int> shared_memory_query_count {0};
  inline static std::atomic<int> load_count {0};
  inline static std::atomic<int> launch_config_query_count {0};
  inline static std::atomic<int> warp_size_query_count {0};
  inline static std::atomic<int> prepare_launch_count {0};
  inline static std::atomic<int> launch_count {0};
  inline static std::atomic<unsigned int> warp_size_value {64};
  inline static std::atomic<bool> cooperative_value {true};
  inline static std::atomic<unsigned int> launched_block_x {0};
  inline static std::atomic<unsigned int> launched_shared_memory {0};
  inline static std::atomic<bool> launched_cooperative {false};

  static void ensure_context() {
  }

  static int get_device_index() {
    return 0;
  }

  static KernelHandle load_kernel(const std::string&, const std::string&) {
    ++load_count;
    return 2;
  }

  static unsigned int get_shared_memory(const std::string&, const std::string&) {
    ++shared_memory_query_count;
    return 192;
  }

  static unsigned int get_warp_size(const std::string&, const std::string&) {
    ++warp_size_query_count;
    return warp_size_value;
  }

  static LaunchOptions prepare_launch(const std::string&,
                                      const std::string&,
                                      unsigned int shared_memory,
                                      const std::string&,
                                      size_t) {
    ++prepare_launch_count;
    return {.shared_memory = shared_memory, .cooperative = cooperative_value.load()};
  }

  static triton_jit::KernelLaunchConfig<LaunchOptions> make_kernel_launch_config(
      const std::string&,
      const std::string&,
      unsigned int shared_memory) {
    ++launch_config_query_count;
    return {
        .warp_size = warp_size_value.load(),
        .options = {.shared_memory = shared_memory, .cooperative = cooperative_value.load()},
    };
  }

  static void launch_kernel(StreamType,
                            KernelHandle,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            unsigned int block_x,
                            unsigned int,
                            unsigned int,
                            void**,
                            const LaunchOptions& opts) {
    ++launch_count;
    launched_block_x = block_x;
    launched_shared_memory = opts.shared_memory;
    launched_cooperative = opts.cooperative;
  }

  static void reset() {
    shared_memory_query_count = 0;
    load_count = 0;
    launch_config_query_count = 0;
    warp_size_query_count = 0;
    prepare_launch_count = 0;
    launch_count = 0;
    warp_size_value = 64;
    cooperative_value = true;
    launched_block_x = 0;
    launched_shared_memory = 0;
    launched_cooperative = false;
  }
};

static_assert(triton_jit::BackendPolicy<CachedLaunchBackend>);
static_assert(triton_jit::KernelInvariantLaunchConfigBackend<CachedLaunchBackend>);

void test_shared_memory_is_cached_at_construction() {
  using FakeKernel = triton_jit::TritonKernelImpl<FakeBackend>;

  FakeBackend::shared_memory_query_count = 0;
  FakeBackend::load_count = 0;
  FakeBackend::prepare_launch_count = 0;
  FakeBackend::launch_count = 0;
  FakeBackend::shared_memory_value = 128;
  FakeBackend::prepared_shared_memory = 0;
  FakeBackend::launched_shared_memory = 0;

  FakeKernel kernel("unused", "fake_kernel");
  require(FakeBackend::shared_memory_query_count == 1,
          "shared memory must be queried exactly once when the kernel entry is constructed");
  require(FakeBackend::load_count == 0, "constructing a kernel entry must preserve lazy module loading");

  // A later backend value change must not affect this compiled-kernel entry.
  FakeBackend::shared_memory_value = 256;
  kernel.launch(1, 1, 1, 1, nullptr, nullptr);
  kernel.launch(1, 1, 1, 1, nullptr, nullptr);

  require(FakeBackend::shared_memory_query_count == 1,
          "launches must reuse the shared-memory value cached by the kernel entry");
  require(FakeBackend::load_count == 1, "the kernel module must still be loaded only once");
  require(FakeBackend::launch_count == 2, "both launches must reach the backend");
  require(FakeBackend::prepare_launch_count == 2,
          "backends without invariant launch configs must keep per-launch preparation");
  require(FakeBackend::prepared_shared_memory == 128,
          "prepare_launch must receive the cached shared-memory value");
  require(FakeBackend::launched_shared_memory == 128,
          "launch_kernel must receive the cached shared-memory value");
}

void test_kernel_invariant_launch_config_is_cached_after_loading() {
  using CachedKernel = triton_jit::TritonKernelImpl<CachedLaunchBackend>;

  CachedLaunchBackend::reset();
  CachedKernel kernel("unused", "cached_launch_kernel");

  require(CachedLaunchBackend::shared_memory_query_count == 1,
          "shared memory must still be cached when the kernel entry is constructed");
  require(CachedLaunchBackend::load_count == 0,
          "launch-config caching must preserve lazy module loading");
  require(CachedLaunchBackend::launch_config_query_count == 0,
          "launch config must not be queried before the module is loaded");

  kernel.launch_with_signature(1, 1, 1, 2, nullptr, nullptr, "first", 3);

  require(CachedLaunchBackend::load_count == 1, "the first launch must load the module once");
  require(CachedLaunchBackend::launch_config_query_count == 1,
          "the first launch must prepare one invariant launch config");
  require(CachedLaunchBackend::warp_size_query_count == 0,
          "the hot path must not query warp size separately");
  require(CachedLaunchBackend::prepare_launch_count == 0,
          "the hot path must not prepare per-call launch options");
  require(CachedLaunchBackend::launched_block_x == 128,
          "the launch must use the warp size from the cached config");
  require(CachedLaunchBackend::launched_shared_memory == 192,
          "the launch must use cached shared memory");
  require(CachedLaunchBackend::launched_cooperative,
          "the launch must use the cached backend option");

  CachedLaunchBackend::warp_size_value = 32;
  CachedLaunchBackend::cooperative_value = false;
  kernel.launch_with_signature(1, 1, 1, 3, nullptr, nullptr, "second", 7);

  require(CachedLaunchBackend::load_count == 1,
          "later launches must reuse the loaded module");
  require(CachedLaunchBackend::launch_config_query_count == 1,
          "later launches must reuse the invariant launch config");
  require(CachedLaunchBackend::warp_size_query_count == 0,
          "later launches must not query warp size separately");
  require(CachedLaunchBackend::prepare_launch_count == 0,
          "later launches must not prepare launch options again");
  require(CachedLaunchBackend::launch_count == 2, "both launches must reach the backend");
  require(CachedLaunchBackend::launched_block_x == 192,
          "later launches must retain the cached warp size");
  require(CachedLaunchBackend::launched_cooperative,
          "later launches must retain the cached backend option");
}

void test_kernel_invariant_launch_config_is_published_once() {
  using CachedKernel = triton_jit::TritonKernelImpl<CachedLaunchBackend>;
  constexpr int kThreadCount = 16;

  CachedLaunchBackend::reset();
  CachedKernel kernel("unused", "concurrent_cached_launch_kernel");
  std::barrier start(kThreadCount);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int index = 0; index < kThreadCount; ++index) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      kernel.launch(1, 1, 1, 2, nullptr, nullptr);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  require(CachedLaunchBackend::load_count == 1,
          "concurrent first launches must load the module exactly once");
  require(CachedLaunchBackend::launch_config_query_count == 1,
          "concurrent first launches must prepare the cached config exactly once");
  require(CachedLaunchBackend::launch_count == kThreadCount,
          "every concurrent launch must use the published config");
  require(CachedLaunchBackend::launched_block_x == 128,
          "concurrent launches must observe the initialized warp size");
  require(CachedLaunchBackend::launched_shared_memory == 192,
          "concurrent launches must observe the initialized launch options");
}

}  // namespace

int main() {
  try {
    test_shared_memory_is_cached_at_construction();
    test_kernel_invariant_launch_config_is_cached_after_loading();
    test_kernel_invariant_launch_config_is_published_once();
    std::cout << "shared-memory cache tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
