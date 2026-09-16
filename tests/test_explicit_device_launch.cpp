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

#include "triton_jit/triton_jit_function.h"

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

struct FakeBackend {
  using StreamType = void*;
  using ContextType = void*;
  using KernelHandle = int;
  using LaunchOptions = int;

  static constexpr unsigned int WARP_SIZE = 32;

  inline static int ensure_context_count = 0;
  inline static int get_device_index_count = 0;
  inline static int load_count = 0;
  inline static int launch_count = 0;
  inline static int current_device_index = 0;
  inline static std::map<int, int> cache_entry_count_by_device;

  static void ensure_context() {
    ++ensure_context_count;
  }

  static int get_device_index() {
    ++get_device_index_count;
    return current_device_index;
  }

  static KernelHandle load_kernel(const std::string&, const std::string&) {
    ++load_count;
    return load_count;
  }

  static unsigned int get_shared_memory(const std::string&, const std::string&) {
    return 0;
  }

  static LaunchOptions prepare_launch(
      const std::string&, const std::string&, unsigned int, const std::string&, size_t) {
    return 0;
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
                            const LaunchOptions&) {
    ++launch_count;
  }

  static void reset(int device_index = 0) {
    ensure_context_count = 0;
    get_device_index_count = 0;
    load_count = 0;
    launch_count = 0;
    current_device_index = device_index;
    cache_entry_count_by_device.clear();
  }

  static int cache_entries_for(int device_index) {
    auto it = cache_entry_count_by_device.find(device_index);
    return it == cache_entry_count_by_device.end() ? 0 : it->second;
  }
};

static_assert(triton_jit::BackendPolicy<FakeBackend>);

namespace triton_jit {

template <>
TritonJITFunctionImpl<FakeBackend>::TritonJITFunctionImpl(std::string_view path, std::string_view name)
    : file_path_(path), function_name_(name), static_sig_ {1, {ArgType::NON_CONSTEXPR}} {
}

template <>
const TritonKernelImpl<FakeBackend>& TritonJITFunctionImpl<FakeBackend>::get_kernel(
    std::string_view signature, const CompileOptions& opts, int device_index) const {
  std::string key = detail::make_kernel_cache_key(signature, device_index, opts);
  return overloads_->get_or_create(std::move(key), [device_index](const std::string&) {
    ++FakeBackend::cache_entry_count_by_device[device_index];
    return std::make_unique<TritonKernelImpl<FakeBackend>>("device-" + std::to_string(device_index),
                                                           "fake_kernel");
  });
}

}  // namespace triton_jit

namespace {

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    throw std::runtime_error("check failed at line " + std::to_string(line) + ": " + expression);
  }
}

#define REQUIRE(expression) check(static_cast<bool>(expression), #expression, __LINE__)

using FakeFunction = triton_jit::TritonJITFunctionImpl<FakeBackend>;

void test_explicit_typed_launch_skips_backend_device_lookup() {
  FakeBackend::reset();
  FakeFunction& function = FakeFunction::get_instance("explicit_typed_launch.py", "explicit_typed_kernel");

  function.launch_on_device(3, nullptr, 1, 1, 1, 4, 3, 42);

  REQUIRE(FakeBackend::ensure_context_count == 0);
  REQUIRE(FakeBackend::get_device_index_count == 0);
  REQUIRE(FakeBackend::cache_entries_for(3) == 1);
  REQUIRE(FakeBackend::launch_count == 1);
}

void test_legacy_typed_launch_uses_backend_device_lookup() {
  FakeBackend::reset(5);
  FakeFunction& function = FakeFunction::get_instance("legacy_typed_launch.py", "legacy_typed_kernel");

  function(nullptr, 1, 1, 1, 4, 3, 42);

  REQUIRE(FakeBackend::ensure_context_count == 1);
  REQUIRE(FakeBackend::get_device_index_count == 1);
  REQUIRE(FakeBackend::cache_entries_for(5) == 1);
  REQUIRE(FakeBackend::launch_count == 1);
}

void test_explicit_raw_launch_skips_backend_device_lookup() {
  FakeBackend::reset();
  FakeFunction& function = FakeFunction::get_instance("explicit_raw_launch.py", "explicit_raw_kernel");
  int argument = 42;
  void* arguments[] = {&argument};

  function.launch_with_raw_args_on_device(7, nullptr, 1, 1, 1, 4, 3, "i32", arguments, 1);

  REQUIRE(FakeBackend::ensure_context_count == 0);
  REQUIRE(FakeBackend::get_device_index_count == 0);
  REQUIRE(FakeBackend::cache_entries_for(7) == 1);
  REQUIRE(FakeBackend::launch_count == 1);
}

void test_legacy_raw_launch_uses_backend_device_lookup() {
  FakeBackend::reset(9);
  FakeFunction& function = FakeFunction::get_instance("legacy_raw_launch.py", "legacy_raw_kernel");
  int argument = 42;
  void* arguments[] = {&argument};

  function.launch_with_raw_args(nullptr, 1, 1, 1, 4, 3, "i32", arguments, 1);

  REQUIRE(FakeBackend::ensure_context_count == 1);
  REQUIRE(FakeBackend::get_device_index_count == 1);
  REQUIRE(FakeBackend::cache_entries_for(9) == 1);
  REQUIRE(FakeBackend::launch_count == 1);
}

void test_explicit_devices_use_separate_cache_entries() {
  FakeBackend::reset();
  FakeFunction& function =
      FakeFunction::get_instance("explicit_device_cache.py", "explicit_device_cache_kernel");

  function.launch_on_device(3, nullptr, 1, 1, 1, 4, 3, 42);
  function.launch_on_device(3, nullptr, 1, 1, 1, 4, 3, 42);
  function.launch_on_device(7, nullptr, 1, 1, 1, 4, 3, 42);

  REQUIRE(FakeBackend::ensure_context_count == 0);
  REQUIRE(FakeBackend::get_device_index_count == 0);
  REQUIRE(FakeBackend::cache_entry_count_by_device.size() == 2);
  REQUIRE(FakeBackend::cache_entries_for(3) == 1);
  REQUIRE(FakeBackend::cache_entries_for(7) == 1);
  REQUIRE(FakeBackend::load_count == 2);
  REQUIRE(FakeBackend::launch_count == 3);
}

}  // namespace

int main() {
  try {
    test_explicit_typed_launch_skips_backend_device_lookup();
    test_legacy_typed_launch_uses_backend_device_lookup();
    test_explicit_raw_launch_skips_backend_device_lookup();
    test_legacy_raw_launch_uses_backend_device_lookup();
    test_explicit_devices_use_separate_cache_entries();
    std::cout << "explicit-device launch tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
