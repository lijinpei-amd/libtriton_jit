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

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace triton_jit::detail {

// An append-only cache with stable object addresses. Cache misses are created
// outside the lock: callers may acquire unrelated locks (notably Python's GIL)
// without introducing a cache-lock/GIL lock-order inversion. Concurrent misses
// may create duplicate candidates, but only one candidate is published.
template <typename Key,
          typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class ThreadSafeCache {
 public:
  ThreadSafeCache() = default;

  ThreadSafeCache(const ThreadSafeCache&) = delete;
  ThreadSafeCache& operator=(const ThreadSafeCache&) = delete;
  ThreadSafeCache(ThreadSafeCache&&) = delete;
  ThreadSafeCache& operator=(ThreadSafeCache&&) = delete;

  Value* find(const Key& key) {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.get();
  }

  const Value* find(const Key& key) const {
    std::shared_lock lock(mutex_);
    auto it = entries_.find(key);
    return it == entries_.end() ? nullptr : it->second.get();
  }

  template <typename Factory>
  Value& get_or_create(Key key, Factory&& factory) {
    if (Value* cached = find(key)) {
      return *cached;
    }

    // Deliberately run user work without holding mutex_. Besides keeping hits
    // for other keys moving, this is required when the factory acquires the GIL.
    std::unique_ptr<Value> candidate = std::forward<Factory>(factory)(key);
    if (!candidate) {
      throw std::invalid_argument("ThreadSafeCache factory returned null");
    }

    std::unique_lock lock(mutex_);
    auto result = entries_.try_emplace(std::move(key), std::move(candidate));
    return *result.first->second;
  }

  size_t size() const {
    std::shared_lock lock(mutex_);
    return entries_.size();
  }

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<Key, std::unique_ptr<Value>, Hash, KeyEqual> entries_;
};

}  // namespace triton_jit::detail
