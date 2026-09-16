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
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "triton_jit/triton_jit_function.h"

namespace {

static_assert(std::is_same_v<
              decltype(std::declval<triton_jit::ParameterBuffer&>().get_ptrs()),
              std::span<void*>>);

bool expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

template <size_t Count>
bool test_abi_value_count() {
  triton_jit::ParameterBuffer buffer;
  buffer.reserve(Count);
  for (size_t i = 0; i < Count; ++i) {
    buffer.push_arg(static_cast<uint64_t>(0x1000U + i));
  }

  auto pointers = buffer.get_ptrs();
  bool ok = expect(pointers.size() == Count,
                   "ABI pointer count must match the number of pushed values");
  for (size_t i = 0; i < pointers.size(); ++i) {
    const auto actual = *reinterpret_cast<const uint64_t*>(pointers[i]);
    ok &= expect(actual == 0x1000U + i, "ABI pointer must address the pushed value");
  }
  return ok;
}

struct alignas(16) AlignedValue {
  uint64_t low;
  uint64_t high;
};

bool test_alignment() {
  triton_jit::ParameterBuffer buffer;
  buffer.reserve(4);
  buffer.push_arg(uint8_t{0x12});
  buffer.push_arg(uint32_t{0x3456789A});
  buffer.push_arg(uint64_t{0x0123456789ABCDEFULL});
  buffer.push_arg(AlignedValue{0x1111222233334444ULL, 0x5555666677778888ULL});

  auto pointers = buffer.get_ptrs();
  bool ok = expect(pointers.size() == 4, "mixed values must produce four ABI pointers");
  if (pointers.size() != 4) {
    return false;
  }

  const std::array<size_t, 4> alignments{
      alignof(uint8_t), alignof(uint32_t), alignof(uint64_t), alignof(AlignedValue)};
  for (size_t i = 0; i < pointers.size(); ++i) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointers[i]);
    ok &= expect(address % alignments[i] == 0,
                 "each ABI pointer must satisfy its value's alignment");
  }

  ok &= expect(*reinterpret_cast<const uint8_t*>(pointers[0]) == uint8_t{0x12},
               "uint8_t value must be preserved");
  ok &= expect(*reinterpret_cast<const uint32_t*>(pointers[1]) == uint32_t{0x3456789A},
               "uint32_t value must be preserved");
  ok &= expect(*reinterpret_cast<const uint64_t*>(pointers[2]) ==
                   uint64_t{0x0123456789ABCDEFULL},
               "uint64_t value must be preserved");
  const auto& aligned = *reinterpret_cast<const AlignedValue*>(pointers[3]);
  ok &= expect(aligned.low == 0x1111222233334444ULL &&
                   aligned.high == 0x5555666677778888ULL,
               "16-byte-aligned value must be preserved");
  return ok;
}

bool test_repeated_get_ptrs() {
  triton_jit::ParameterBuffer buffer;
  buffer.reserve(5);
  buffer.push_arg(uint32_t{10});
  buffer.push_arg(uint32_t{20});
  buffer.push_arg(uint32_t{30});
  buffer.push_arg(uint32_t{40});

  auto first = buffer.get_ptrs();
  bool ok = expect(first.size() == 4, "first pointer view must contain four entries");
  void** pointer_storage = first.data();
  std::array<void*, 4> first_addresses{};
  if (first.size() == first_addresses.size()) {
    for (size_t i = 0; i < first.size(); ++i) {
      first_addresses[i] = first[i];
    }
  }

  auto second = buffer.get_ptrs();
  ok &= expect(second.size() == 4, "repeated pointer view must not duplicate entries");
  ok &= expect(second.data() == pointer_storage,
               "repeated pointer view must reuse pointer-array storage");
  if (second.size() == first_addresses.size()) {
    for (size_t i = 0; i < second.size(); ++i) {
      ok &= expect(second[i] == first_addresses[i],
                   "repeated pointer view must address the same values");
      ok &= expect(*reinterpret_cast<const uint32_t*>(second[i]) == (i + 1) * 10,
                   "repeated pointer view must preserve values");
    }
  }

  buffer.push_arg(uint32_t{50});
  auto third = buffer.get_ptrs();
  ok &= expect(third.size() == 5, "pointer view must rebuild after another argument is added");
  ok &= expect(third.data() == pointer_storage,
               "reserved pointer-array storage must survive another argument");
  if (third.size() == 5) {
    for (size_t i = 0; i < third.size(); ++i) {
      ok &= expect(*reinterpret_cast<const uint32_t*>(third[i]) == (i + 1) * 10,
                   "rebuilt pointer view must preserve all values");
    }
  }
  return ok;
}

bool test_tuple_expansion() {
  using namespace triton_jit;

  StaticSignature static_signature{1, {ArgType::NON_CONSTEXPR}};
  ParameterBuffer buffer;
  buffer.reserve(3);
  c10::SmallVector<std::string> signature;
  ArgHandle handler{static_signature, buffer, signature, 0};

  handler.handle_arg(std::tuple<uint32_t, uint64_t, float>{7U, 11U, 13.5F});
  auto pointers = buffer.get_ptrs();

  bool ok = expect(handler.idx == 1, "tuple must consume one static-signature slot");
  ok &= expect(buffer.size() == 3, "tuple must expand to one ABI value per element");
  ok &= expect(signature.size() == 1, "tuple must emit one grouped signature token");
  if (signature.size() == 1) {
    ok &= expect(signature[0] == "(u32,u64,fp32)",
                 "tuple signature must preserve element types");
  }
  ok &= expect(pointers.size() == 3, "tuple must expose three ABI pointers");
  if (pointers.size() == 3) {
    ok &= expect(*reinterpret_cast<const uint32_t*>(pointers[0]) == 7U,
                 "tuple element 0 must be preserved");
    ok &= expect(*reinterpret_cast<const uint64_t*>(pointers[1]) == 11U,
                 "tuple element 1 must be preserved");
    ok &= expect(*reinterpret_cast<const float*>(pointers[2]) == 13.5F,
                 "tuple element 2 must be preserved");
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_abi_value_count<0>();
  ok &= test_abi_value_count<4>();
  ok &= test_abi_value_count<11>();
  ok &= test_abi_value_count<32>();
  ok &= test_abi_value_count<33>();
  ok &= test_alignment();
  ok &= test_repeated_get_ptrs();
  ok &= test_tuple_expansion();
  return ok ? 0 : 1;
}
