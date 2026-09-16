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

#include <iostream>
#include <stdexcept>
#include <string>

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

void test_shared_memory_is_cached_at_construction() {
  using FakeKernel = triton_jit::TritonKernelImpl<FakeBackend>;

  FakeBackend::shared_memory_query_count = 0;
  FakeBackend::load_count = 0;
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
  require(FakeBackend::prepared_shared_memory == 128,
          "prepare_launch must receive the cached shared-memory value");
  require(FakeBackend::launched_shared_memory == 128,
          "launch_kernel must receive the cached shared-memory value");
}

}  // namespace

int main() {
  try {
    test_shared_memory_is_cached_at_construction();
    std::cout << "shared-memory cache tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
