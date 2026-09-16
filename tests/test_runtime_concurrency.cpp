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

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "triton_jit/thread_safe_cache.h"
#include "triton_jit/triton_jit_function.h"
#include "triton_jit/triton_kernel.h"

namespace {

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    throw std::runtime_error("check failed at line " + std::to_string(line) + ": " + expression);
  }
}

#define REQUIRE(expression) check(static_cast<bool>(expression), #expression, __LINE__)

struct FakeBackend {
  using StreamType = void*;
  using ContextType = void*;
  using KernelHandle = int;
  using LaunchOptions = int;

  static constexpr unsigned int WARP_SIZE = 32;
  static constexpr KernelHandle EXPECTED_HANDLE = 73;

  inline static std::atomic<int> load_count {0};
  inline static std::atomic<int> launch_count {0};
  inline static std::atomic<int> invalid_handle_count {0};
  inline static std::atomic<bool> fail_next_load {false};

  static void launch_kernel(StreamType,
                            KernelHandle kernel,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            void**,
                            const LaunchOptions&) {
    if (kernel != EXPECTED_HANDLE) {
      ++invalid_handle_count;
    }
    ++launch_count;
  }

  static void ensure_context() {
  }

  static int get_device_index() {
    return 0;
  }

  static KernelHandle load_kernel(const std::string&, const std::string&) {
    ++load_count;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (fail_next_load.exchange(false)) {
      throw std::runtime_error("injected load failure");
    }
    return EXPECTED_HANDLE;
  }

  static unsigned int get_shared_memory(const std::string&, const std::string&) {
    return 0;
  }

  static LaunchOptions prepare_launch(
      const std::string&, const std::string&, unsigned int, const std::string&, size_t) {
    return 0;
  }

  static void reset() {
    load_count = 0;
    launch_count = 0;
    invalid_handle_count = 0;
    fail_next_load = false;
  }
};

static_assert(triton_jit::BackendPolicy<FakeBackend>);
static_assert(triton_jit::is_triton_jit_function_move_constructible_v<FakeBackend>);
static_assert(triton_jit::is_triton_kernel_move_constructible_v<FakeBackend>);

using FakeKernel = triton_jit::TritonKernelImpl<FakeBackend>;
using FakeKernelCache = triton_jit::detail::ThreadSafeCache<std::string, FakeKernel>;

void test_same_key_has_one_published_object() {
  constexpr size_t kThreadCount = 24;
  FakeKernelCache cache;
  std::barrier<> start(static_cast<std::ptrdiff_t>(kThreadCount));
  std::atomic<int> factory_count {0};
  std::array<const FakeKernel*, kThreadCount> results {};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      start.arrive_and_wait();
      results[i] = &cache.get_or_create("same", [&](const std::string& key) {
        ++factory_count;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::make_unique<FakeKernel>("unused", key);
      });
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(cache.size() == 1);
  REQUIRE(factory_count.load() >= 1);
  for (const FakeKernel* result : results) {
    REQUIRE(result == results.front());
  }
}

void test_factory_runs_outside_cache_lock() {
  FakeKernelCache cache;
  const FakeKernel& outer = cache.get_or_create("outer", [&](const std::string& key) {
    const FakeKernel& inner = cache.get_or_create("inner", [](const std::string& inner_key) {
      return std::make_unique<FakeKernel>("unused", inner_key);
    });
    REQUIRE(inner.get_kernel_name() == "inner");
    return std::make_unique<FakeKernel>("unused", key);
  });

  REQUIRE(outer.get_kernel_name() == "outer");
  REQUIRE(cache.size() == 2);
}

void test_factory_failure_does_not_poison_cache() {
  FakeKernelCache cache;
  bool caught = false;
  try {
    (void)cache.get_or_create("retry", [](const std::string&) -> std::unique_ptr<FakeKernel> {
      throw std::runtime_error("injected factory failure");
    });
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "injected factory failure";
  }

  REQUIRE(caught);
  REQUIRE(cache.size() == 0);
  const FakeKernel& kernel = cache.get_or_create("retry", [](const std::string& key) {
    return std::make_unique<FakeKernel>("unused", key);
  });
  REQUIRE(kernel.get_kernel_name() == "retry");
  REQUIRE(cache.size() == 1);
}

void test_rehash_keeps_published_objects_stable() {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kKeysPerThread = 128;
  FakeKernelCache cache;
  const FakeKernel* anchor = &cache.get_or_create("anchor", [](const std::string& key) {
    return std::make_unique<FakeKernel>("unused", key);
  });
  std::atomic<bool> valid {true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      for (size_t key_index = 0; key_index < kKeysPerThread; ++key_index) {
        const std::string key = "key-" + std::to_string(thread_index) + "-" + std::to_string(key_index);
        const FakeKernel& kernel = cache.get_or_create(key, [](const std::string& candidate_key) {
          return std::make_unique<FakeKernel>("unused", candidate_key);
        });
        if (kernel.get_kernel_name() != key) {
          valid = false;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(valid.load());
  REQUIRE(cache.size() == 1 + kThreadCount * kKeysPerThread);
  REQUIRE(cache.find("anchor") == anchor);
}

void test_kernel_handle_is_loaded_once() {
  constexpr size_t kThreadCount = 24;
  FakeBackend::reset();
  FakeKernel kernel("unused", "concurrent_kernel");
  std::barrier<> start(static_cast<std::ptrdiff_t>(kThreadCount));
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&]() {
      start.arrive_and_wait();
      kernel.launch(1, 1, 1, 1, nullptr, nullptr);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(kernel.is_loaded());
  REQUIRE(FakeBackend::load_count.load() == 1);
  REQUIRE(FakeBackend::launch_count.load() == static_cast<int>(kThreadCount));
  REQUIRE(FakeBackend::invalid_handle_count.load() == 0);
}

void test_failed_kernel_load_can_retry() {
  FakeBackend::reset();
  FakeBackend::fail_next_load = true;
  FakeKernel kernel("unused", "retry_kernel");

  bool caught = false;
  try {
    kernel.launch(1, 1, 1, 1, nullptr, nullptr);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "injected load failure";
  }

  REQUIRE(caught);
  REQUIRE(!kernel.is_loaded());
  kernel.launch(1, 1, 1, 1, nullptr, nullptr);
  REQUIRE(kernel.is_loaded());
  REQUIRE(FakeBackend::load_count.load() == 2);
  REQUIRE(FakeBackend::launch_count.load() == 1);
  REQUIRE(FakeBackend::invalid_handle_count.load() == 0);
}

}  // namespace

int main() {
  try {
    test_same_key_has_one_published_object();
    test_factory_runs_outside_cache_lock();
    test_factory_failure_does_not_poison_cache();
    test_rehash_keeps_published_objects_stable();
    test_kernel_handle_is_loaded_once();
    test_failed_kernel_load_can_retry();
  } catch (const std::exception& error) {
    std::cerr << "runtime concurrency test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "runtime concurrency tests passed\n";
  return 0;
}
