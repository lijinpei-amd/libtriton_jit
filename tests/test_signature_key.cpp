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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/signature_key.h"
#include "triton_jit/triton_jit_function.h"

struct LegacyOnlyScalar {
  uint32_t value;
};

template <>
struct triton_jit::triton_type<LegacyOnlyScalar> {
  static constexpr const char* name = "legacy_scalar";
};

template <>
struct fmt::formatter<LegacyOnlyScalar> {
  constexpr auto parse(fmt::format_parse_context& context) {
    return context.begin();
  }

  template <typename FormatContext>
  auto format(const LegacyOnlyScalar& value, FormatContext& context) const {
    return fmt::format_to(context.out(), "{}", value.value);
  }
};

namespace {

std::atomic<size_t> tracked_allocations = 0;
thread_local bool track_allocations = false;

void* allocate_memory(size_t size) {
  if (track_allocations) {
    tracked_allocations.fetch_add(1, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

}  // namespace

void* operator new(size_t size) {
  return allocate_memory(size);
}

void* operator new[](size_t size) {
  return allocate_memory(size);
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, size_t) noexcept {
  std::free(pointer);
}

namespace {

using triton_jit::ArgType;
using triton_jit::CompileOptions;
using triton_jit::ParameterBuffer;
using triton_jit::StaticSignature;
using triton_jit::TritonDevicePtr;
using triton_jit::TritonDType;
using triton_jit::detail::KernelCacheKey;
using triton_jit::detail::KernelCacheKeyHash;
using triton_jit::detail::make_kernel_cache_key;
using triton_jit::detail::render_signature;
using triton_jit::detail::SignatureKey;
using triton_jit::detail::SignatureSpecialization;
using triton_jit::detail::StructuralArgHandle;

bool expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

template <typename Exception, typename Function>
bool expect_throws(Function&& function, std::string_view message) {
  try {
    std::forward<Function>(function)();
  } catch (const Exception&) {
    return true;
  } catch (...) {
  }
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

std::string specialization_suffix(SignatureSpecialization specialization) {
  switch (specialization) {
    case SignatureSpecialization::kNone:
      return {};
    case SignatureSpecialization::kDivisibleBy16:
      return ":16";
    case SignatureSpecialization::kEqualToOne:
      return ":1";
  }
  return "<invalid>";
}

struct SignatureSample {
  SignatureKey key;
  std::string old_signature;
  bool require_reverse_partition;
};

void add_sample(std::vector<SignatureSample>& samples,
                SignatureKey key,
                std::string old_signature,
                bool require_reverse_partition = true) {
  samples.push_back({std::move(key), std::move(old_signature), require_reverse_partition});
}

struct ArgHandleSample {
  SignatureKey key;
  std::string legacy_signature;
  std::string label;
};

template <typename... Args>
SignatureKey emit_arg_handle_signature(const std::vector<ArgType>& kinds, Args&&... args) {
  StaticSignature static_signature {static_cast<int>(kinds.size()), kinds};
  ParameterBuffer buffer;
  SignatureKey signature;
  StructuralArgHandle handler {static_signature, buffer, signature, 0};
  handler.handle_args(std::forward<Args>(args)...);
  if (handler.idx != static_cast<int>(kinds.size())) {
    throw std::logic_error("ArgHandle did not consume the complete static signature");
  }
  return signature;
}

template <typename... Args>
size_t emit_arg_handle_abi_count(const std::vector<ArgType>& kinds, Args&&... args) {
  StaticSignature static_signature {static_cast<int>(kinds.size()), kinds};
  ParameterBuffer buffer;
  SignatureKey signature;
  StructuralArgHandle handler {static_signature, buffer, signature, 0};
  handler.handle_args(std::forward<Args>(args)...);
  return buffer.size();
}

template <typename... Args>
size_t add_arg_handle_sample(std::vector<ArgHandleSample>& samples,
                             const std::vector<ArgType>& kinds,
                             std::string legacy_signature,
                             std::string label,
                             Args&&... args) {
  samples.push_back({emit_arg_handle_signature(kinds, std::forward<Args>(args)...),
                     std::move(legacy_signature),
                     std::move(label)});
  return samples.size() - 1;
}

std::string_view arg_type_name(ArgType kind) {
  switch (kind) {
    case ArgType::NON_CONSTEXPR:
      return "non_constexpr";
    case ArgType::SPECIALIZED:
      return "specialized";
    case ArgType::CONSTEXPR:
      return "constexpr";
    case ArgType::SPECIALIZED_NO_ALIGNMENT:
      return "specialized_no_alignment";
  }
  return "invalid";
}

template <typename T>
constexpr std::string_view legacy_declared_type_name() {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, bool>) {
    return "i1";
  } else if constexpr (std::is_same_v<U, int>) {
    return "i32";
  } else if constexpr (std::is_same_v<U, unsigned int>) {
    return "u32";
  } else if constexpr (std::is_same_v<U, int64_t>) {
    return "i64";
  } else if constexpr (std::is_same_v<U, uint64_t>) {
    return "u64";
  } else if constexpr (std::is_same_v<U, float>) {
    return "fp32";
  } else if constexpr (std::is_same_v<U, double>) {
    return "fp64";
  } else {
    static_assert(!sizeof(U), "unsupported scalar type in legacy signature oracle");
  }
}

template <typename T>
std::string legacy_runtime_type_name(const T& value) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
    if (value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max()) {
      return "i32";
    }
    return "i64";
  }
  return std::string(legacy_declared_type_name<T>());
}

template <typename T>
std::string legacy_scalar_signature(ArgType kind, const T& value) {
  if (kind == ArgType::CONSTEXPR) {
    return fmt::format("{}", value);
  }

  std::string signature = legacy_runtime_type_name(value);
  if constexpr (std::is_integral_v<std::remove_cvref_t<T>>) {
#if !defined(BACKEND_NPU)
    if (kind == ArgType::SPECIALIZED) {
      signature += value % 16 == 0 ? ":16" : value == 1 ? ":1" : "";
    } else if (kind == ArgType::SPECIALIZED_NO_ALIGNMENT && value == 1) {
      signature += ":1";
    }
#endif
  }
  return signature;
}

std::string legacy_pointer_signature(ArgType kind, const TritonDevicePtr& pointer) {
  std::string signature = "*";
  signature += triton_jit::to_triton_typename(pointer.dtype);
#if !defined(BACKEND_NPU)
  if (kind == ArgType::SPECIALIZED && pointer.value % 16 == 0) {
    signature += ":16";
  }
#endif
  return signature;
}

template <typename... Ts>
std::string legacy_tuple_signature(const std::tuple<Ts...>&) {
  std::string signature = "(";
  bool first = true;
  auto append_type = [&]<typename T>() {
    if (!first) {
      signature.push_back(',');
    }
    first = false;
    signature += legacy_declared_type_name<T>();
  };
  (append_type.template operator()<Ts>(), ...);
  signature.push_back(')');
  return signature;
}

bool test_signature_partition_and_rendering() {
  constexpr std::array all_dtypes {TritonDType::kI1,
                                   TritonDType::kI8,
                                   TritonDType::kI16,
                                   TritonDType::kI32,
                                   TritonDType::kI64,
                                   TritonDType::kU8,
                                   TritonDType::kU16,
                                   TritonDType::kU32,
                                   TritonDType::kU64,
                                   TritonDType::kFp16,
                                   TritonDType::kBf16,
                                   TritonDType::kFp32,
                                   TritonDType::kFp64,
                                   TritonDType::kFp8E4NV,
                                   TritonDType::kFp8E5};
  constexpr std::array integer_dtypes {TritonDType::kI1,
                                       TritonDType::kI8,
                                       TritonDType::kI16,
                                       TritonDType::kI32,
                                       TritonDType::kI64,
                                       TritonDType::kU8,
                                       TritonDType::kU16,
                                       TritonDType::kU32,
                                       TritonDType::kU64};

  std::vector<SignatureSample> samples;
  for (const TritonDType dtype : all_dtypes) {
    for (int duplicate = 0; duplicate < 2; ++duplicate) {
      SignatureKey scalar;
      scalar.append_runtime_scalar(dtype);
      add_sample(samples, std::move(scalar), triton_jit::to_triton_typename(dtype));

      SignatureKey pointer;
      pointer.append_runtime_pointer(dtype);
      add_sample(samples, std::move(pointer), "*" + std::string(triton_jit::to_triton_typename(dtype)));

      SignatureKey aligned_pointer;
      aligned_pointer.append_runtime_pointer(dtype, SignatureSpecialization::kDivisibleBy16);
      add_sample(samples,
                 std::move(aligned_pointer),
                 "*" + std::string(triton_jit::to_triton_typename(dtype)) + ":16");
    }
  }

  for (const TritonDType dtype : integer_dtypes) {
    for (const SignatureSpecialization specialization :
         {SignatureSpecialization::kDivisibleBy16, SignatureSpecialization::kEqualToOne}) {
      SignatureKey scalar;
      scalar.append_runtime_scalar(dtype, specialization);
      add_sample(samples,
                 std::move(scalar),
                 std::string(triton_jit::to_triton_typename(dtype)) + specialization_suffix(specialization));
    }
  }

  constexpr std::array tuple_types {TritonDType::kFp32, TritonDType::kI32, TritonDType::kU64};
  SignatureKey tuple;
  tuple.append_runtime_tuple(tuple_types);
  add_sample(samples, tuple, "(fp32,i32,u64)");
  add_sample(samples, tuple, "(fp32,i32,u64)");

  SignatureKey tuple_then_scalar;
  tuple_then_scalar.append_runtime_tuple(tuple_types);
  tuple_then_scalar.append_runtime_scalar(TritonDType::kFp64);
  add_sample(samples, std::move(tuple_then_scalar), "(fp32,i32,u64),fp64");

  constexpr std::array second_tuple_types {TritonDType::kI32, TritonDType::kU64};
  SignatureKey scalar_then_tuple;
  scalar_then_tuple.append_runtime_scalar(TritonDType::kFp32);
  scalar_then_tuple.append_runtime_tuple(second_tuple_types);
  add_sample(samples, std::move(scalar_then_tuple), "fp32,(i32,u64)");

  constexpr std::array<int64_t, 11> signed_values {
      std::numeric_limits<int64_t>::min(),
      -17,
      -1,
      0,
      1,
      15,
      16,
      17,
      std::numeric_limits<int32_t>::max(),
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
      std::numeric_limits<int64_t>::max()};
  for (const int64_t value : signed_values) {
    SignatureKey integer;
    integer.append_constexpr_integer(value);
    add_sample(samples, std::move(integer), fmt::format("{}", value));
  }

  constexpr std::array<uint64_t, 7> unsigned_values {
      0,
      1,
      15,
      16,
      17,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      std::numeric_limits<uint64_t>::max()};
  for (const uint64_t value : unsigned_values) {
    SignatureKey integer;
    integer.append_constexpr_integer(value);
    add_sample(samples, std::move(integer), fmt::format("{}", value));
  }

  for (const bool value : {false, true}) {
    SignatureKey boolean;
    boolean.append_constexpr_bool(value);
    add_sample(samples, std::move(boolean), fmt::format("{}", value));
  }

  for (const float value : {-0.0F, 1.25F, std::numeric_limits<float>::infinity()}) {
    SignatureKey floating;
    floating.append_constexpr_f32(value);
    add_sample(samples, std::move(floating), fmt::format("{}", value), false);
  }
  for (const double value : {-0.0, 1.25, std::numeric_limits<double>::infinity()}) {
    SignatureKey floating;
    floating.append_constexpr_f64(value);
    add_sample(samples, std::move(floating), fmt::format("{}", value), false);
  }

  for (const std::string_view value : {"tl.float32", "", "a,b", "true"}) {
    SignatureKey text;
    text.append_constexpr_string(value);
    add_sample(samples, std::move(text), std::string(value), false);
  }

  SignatureKey nullopt;
  nullopt.append_nullopt();
  add_sample(samples, std::move(nullopt), "nullopt");

  const std::string jit_token = "@jit:2f746d702f6d2e7079:666e:0123456789abcdef";
  SignatureKey jit;
  jit.append_jit_function(jit_token);
  add_sample(samples, std::move(jit), jit_token, false);

  for (const std::string_view raw : {"", "i32", " f32 ", "(i32, fp32)", "unknown-token,,"}) {
    add_sample(samples, SignatureKey::raw_fallback(raw), std::string(raw), false);
  }

  SignatureKey composite;
  composite.append_runtime_pointer(TritonDType::kFp32, SignatureSpecialization::kDivisibleBy16);
  composite.append_runtime_scalar(TritonDType::kI32, SignatureSpecialization::kEqualToOne);
  composite.append_runtime_tuple(tuple_types);
  composite.append_constexpr_integer(-33);
  composite.append_constexpr_bool(true);
  composite.append_constexpr_f32(2.5F);
  composite.append_constexpr_f64(-4.75);
  composite.append_constexpr_string("tl.float16");
  composite.append_nullopt();
  composite.append_jit_function(jit_token);
  add_sample(samples,
             std::move(composite),
             "*fp32:16,i32:1,(fp32,i32,u64),-33,true,2.5,-4.75,tl.float16,nullopt," + jit_token,
             false);

  bool ok = true;
  for (const auto& sample : samples) {
    ok &= expect(render_signature(sample.key) == sample.old_signature,
                 "structural signature must render exactly like the string path");
  }

  for (size_t i = 0; i < samples.size(); ++i) {
    for (size_t j = 0; j < samples.size(); ++j) {
      const bool keys_equal = samples[i].key == samples[j].key;
      const bool strings_equal = samples[i].old_signature == samples[j].old_signature;
      ok &= expect(!keys_equal || strings_equal,
                   "equal structural keys must never merge different string signatures");
      if (samples[i].require_reverse_partition && samples[j].require_reverse_partition) {
        ok &= expect(keys_equal == strings_equal,
                     "canonical atoms must induce the same partition as string signatures");
      }
      if (keys_equal) {
        ok &= expect(samples[i].key.hash() == samples[j].key.hash(),
                     "equal signature keys must have equal cached hashes");
      }
    }
  }

  SignatureKey signed_integer;
  signed_integer.append_constexpr_integer(int64_t {17});
  SignatureKey unsigned_integer;
  unsigned_integer.append_constexpr_integer(uint64_t {17});
  ok &= expect(signed_integer == unsigned_integer,
               "equal positive integers must share a structural key across signedness");

  SignatureKey fp32_one;
  fp32_one.append_constexpr_f32(1.0F);
  SignatureKey fp64_one;
  fp64_one.append_constexpr_f64(1.0);
  ok &= expect(render_signature(fp32_one) == render_signature(fp64_one) && !(fp32_one == fp64_one),
               "different floating widths may conservatively split equal rendered values");

  SignatureKey typed_i32;
  typed_i32.append_runtime_scalar(TritonDType::kI32);
  SignatureKey raw_i32 = SignatureKey::raw_fallback("i32");
  ok &= expect(render_signature(typed_i32) == render_signature(raw_i32) && !(typed_i32 == raw_i32),
               "raw fallbacks must retain an exact, collision-safe identity");
  return ok;
}

bool test_arg_handle_legacy_properties() {
  constexpr std::array scalar_modes {ArgType::NON_CONSTEXPR,
                                     ArgType::SPECIALIZED,
                                     ArgType::CONSTEXPR,
                                     ArgType::SPECIALIZED_NO_ALIGNMENT};
  std::vector<ArgHandleSample> samples;
  bool ok = true;

  auto add_scalar_modes = [&](const auto& value, std::string_view value_label) {
    for (const ArgType mode : scalar_modes) {
      add_arg_handle_sample(samples,
                            {mode},
                            legacy_scalar_signature(mode, value),
                            fmt::format("{} {}", value_label, arg_type_name(mode)),
                            value);
    }
  };

  constexpr std::array<int64_t, 12> signed_values {
      std::numeric_limits<int64_t>::min(),
      static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1,
      std::numeric_limits<int32_t>::min(),
      -17,
      -16,
      -1,
      0,
      1,
      16,
      std::numeric_limits<int32_t>::max(),
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
      std::numeric_limits<int64_t>::max()};
  for (const int64_t value : signed_values) {
    add_scalar_modes(value, fmt::format("int64 {}", value));
  }
  add_scalar_modes(int {-17}, "int -17");
  add_scalar_modes(int {1}, "int 1");
  add_scalar_modes(static_cast<unsigned int>(0), "uint 0");
  add_scalar_modes(static_cast<unsigned int>(1), "uint 1");
  add_scalar_modes(static_cast<unsigned int>(16), "uint 16");
  add_scalar_modes(uint64_t {std::numeric_limits<uint64_t>::max()}, "uint64 max");
  add_scalar_modes(false, "bool false");
  add_scalar_modes(true, "bool true");

  const std::array<float, 7> float_values {0.0F,
                                           -0.0F,
                                           1.25F,
                                           -16.0F,
                                           std::numeric_limits<float>::infinity(),
                                           std::bit_cast<float>(uint32_t {0x7fc00001U}),
                                           std::bit_cast<float>(uint32_t {0x7fc00002U})};
  for (size_t index = 0; index < float_values.size(); ++index) {
    add_scalar_modes(float_values[index], fmt::format("float sample {}", index));
  }
  const std::array<double, 6> double_values {0.0,
                                             -0.0,
                                             1.25,
                                             std::numeric_limits<double>::infinity(),
                                             std::bit_cast<double>(uint64_t {0x7ff8000000000001ULL}),
                                             std::bit_cast<double>(uint64_t {0x7ff8000000000002ULL})};
  for (size_t index = 0; index < double_values.size(); ++index) {
    add_scalar_modes(double_values[index], fmt::format("double sample {}", index));
  }

  for (const int16_t value : {int16_t {-1}, std::numeric_limits<int16_t>::min()}) {
    add_arg_handle_sample(samples,
                          {ArgType::CONSTEXPR},
                          fmt::format("{}", value),
                          fmt::format("narrow signed constexpr {}", value),
                          value);
  }

  for (const ArgType mode : scalar_modes) {
    const int64_t integer_value = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    add_arg_handle_sample(samples,
                          {mode},
                          legacy_scalar_signature(mode, integer_value),
                          fmt::format("c10 integer {}", arg_type_name(mode)),
                          c10::Scalar(integer_value));
    const bool boolean_value = mode != ArgType::NON_CONSTEXPR;
    add_arg_handle_sample(samples,
                          {mode},
                          legacy_scalar_signature(mode, boolean_value),
                          fmt::format("c10 bool {}", arg_type_name(mode)),
                          c10::Scalar(boolean_value));
  }

  constexpr std::array pointer_modes {ArgType::NON_CONSTEXPR,
                                      ArgType::SPECIALIZED,
                                      ArgType::SPECIALIZED_NO_ALIGNMENT};
  constexpr std::array<std::uintptr_t, 4> pointer_values {0, 1, 0x7f0000001000ULL, 0x7f0000001004ULL};
  for (const ArgType mode : pointer_modes) {
    for (const std::uintptr_t address : pointer_values) {
      const TritonDevicePtr pointer {address, TritonDType::kFp32};
      add_arg_handle_sample(samples,
                            {mode},
                            legacy_pointer_signature(mode, pointer),
                            fmt::format("fp32 pointer {} at {}", arg_type_name(mode), address),
                            pointer);
    }
  }
  const TritonDevicePtr fp16_pointer {pointer_values[2], TritonDType::kFp16};
  add_arg_handle_sample(samples,
                        {ArgType::SPECIALIZED},
                        legacy_pointer_signature(ArgType::SPECIALIZED, fp16_pointer),
                        "aligned fp16 pointer",
                        fp16_pointer);

  size_t legacy_only_index = samples.size();
  for (const ArgType mode :
       {ArgType::NON_CONSTEXPR, ArgType::SPECIALIZED, ArgType::SPECIALIZED_NO_ALIGNMENT}) {
    const size_t current_index =
        add_arg_handle_sample(samples,
                              {mode},
                              "legacy_scalar",
                              fmt::format("legacy-only scalar {}", arg_type_name(mode)),
                              LegacyOnlyScalar {7});
    if (legacy_only_index == samples.size() - 1) {
      legacy_only_index = current_index;
    } else {
      ok &= expect(samples[legacy_only_index].key == samples[current_index].key,
                   "legacy trait-only types must retain their textual runtime key");
    }
  }

  at::Tensor tensor = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32));
  const TritonDevicePtr tensor_pointer {reinterpret_cast<std::uintptr_t>(tensor.data_ptr()),
                                        TritonDType::kFp32};
  const size_t tensor_index =
      add_arg_handle_sample(samples,
                            {ArgType::SPECIALIZED},
                            legacy_pointer_signature(ArgType::SPECIALIZED, tensor_pointer),
                            "tensor pointer",
                            tensor);
  const size_t equivalent_pointer_index =
      add_arg_handle_sample(samples,
                            {ArgType::SPECIALIZED},
                            legacy_pointer_signature(ArgType::SPECIALIZED, tensor_pointer),
                            "device pointer equivalent to tensor",
                            tensor_pointer);
  ok &= expect(samples[tensor_index].key == samples[equivalent_pointer_index].key,
               "a tensor and equivalent typed device pointer must share a structural key");

  size_t nullptr_index = samples.size();
  for (const ArgType mode : pointer_modes) {
    const size_t current_index = add_arg_handle_sample(samples,
                                                       {mode},
                                                       "*i8",
                                                       fmt::format("nullptr {}", arg_type_name(mode)),
                                                       nullptr);
    if (nullptr_index == samples.size() - 1) {
      nullptr_index = current_index;
    } else {
      ok &= expect(samples[nullptr_index].key == samples[current_index].key,
                   "runtime nullptr keys must not depend on specialization mode");
    }
  }

  const auto all_types_tuple =
      std::tuple<int, unsigned int, int64_t, uint64_t, float, double> {-3, 4U, -5, 6U, 7.5F, 8.5};
  const auto same_type_tuple =
      std::tuple<int, unsigned int, int64_t, uint64_t, float, double> {30, 40U, 50, 60U, 70.5F, 80.5};
  const size_t tuple_index = add_arg_handle_sample(samples,
                                                   {ArgType::NON_CONSTEXPR},
                                                   legacy_tuple_signature(all_types_tuple),
                                                   "tuple with every supported element type",
                                                   all_types_tuple);
  const size_t same_tuple_index = add_arg_handle_sample(samples,
                                                        {ArgType::SPECIALIZED},
                                                        legacy_tuple_signature(same_type_tuple),
                                                        "same tuple types with different values",
                                                        same_type_tuple);
  ok &= expect(samples[tuple_index].key == samples[same_tuple_index].key,
               "tuple keys must depend on element types, not element values or specialization mode");
  const auto reordered_tuple =
      std::tuple<double, float, uint64_t, int64_t, unsigned int, int> {8.5, 7.5F, 6U, -5, 4U, -3};
  add_arg_handle_sample(samples,
                        {ArgType::NON_CONSTEXPR},
                        legacy_tuple_signature(reordered_tuple),
                        "tuple with reversed element types",
                        reordered_tuple);

  for (const ArgType mode : scalar_modes) {
    const int64_t value = mode == ArgType::SPECIALIZED_NO_ALIGNMENT ? 1 : 16;
    const size_t plain_index =
        add_arg_handle_sample(samples,
                              {mode},
                              legacy_scalar_signature(mode, value),
                              fmt::format("plain optional reference {}", arg_type_name(mode)),
                              value);
    const size_t optional_index =
        add_arg_handle_sample(samples,
                              {mode},
                              legacy_scalar_signature(mode, value),
                              fmt::format("engaged scalar optional {}", arg_type_name(mode)),
                              std::optional<int64_t> {value});
    ok &= expect(samples[plain_index].key == samples[optional_index].key,
                 "an engaged scalar optional must share the plain value's key");
  }

  size_t first_empty_optional = samples.size();
  for (const ArgType mode : scalar_modes) {
    const size_t empty_index =
        add_arg_handle_sample(samples,
                              {mode},
                              "nullopt",
                              fmt::format("empty scalar optional {}", arg_type_name(mode)),
                              std::optional<int64_t> {});
    if (first_empty_optional == samples.size() - 1) {
      first_empty_optional = empty_index;
    } else {
      ok &= expect(samples[first_empty_optional].key == samples[empty_index].key,
                   "empty optionals must have one mode-independent key");
    }
  }
  const TritonDevicePtr optional_pointer {pointer_values[2], TritonDType::kFp32};
  const size_t plain_pointer_index =
      add_arg_handle_sample(samples,
                            {ArgType::SPECIALIZED},
                            legacy_pointer_signature(ArgType::SPECIALIZED, optional_pointer),
                            "plain pointer optional reference",
                            optional_pointer);
  const size_t engaged_pointer_index =
      add_arg_handle_sample(samples,
                            {ArgType::SPECIALIZED},
                            legacy_pointer_signature(ArgType::SPECIALIZED, optional_pointer),
                            "engaged pointer optional",
                            std::optional<TritonDevicePtr> {optional_pointer});
  ok &= expect(samples[plain_pointer_index].key == samples[engaged_pointer_index].key,
               "an engaged pointer optional must share the plain pointer's key");
  const size_t empty_pointer_index = add_arg_handle_sample(samples,
                                                           {ArgType::SPECIALIZED},
                                                           "nullopt",
                                                           "empty pointer optional",
                                                           std::optional<TritonDevicePtr> {});
  ok &= expect(samples[first_empty_optional].key == samples[empty_pointer_index].key,
               "empty optionals of different element types must share the nullopt key");

  const std::string common_text = "tl.float32";
  const std::string_view common_view = common_text;
  const char* common_pointer = common_text.c_str();
  const size_t string_index = add_arg_handle_sample(samples,
                                                    {ArgType::CONSTEXPR},
                                                    fmt::format("{}", common_text),
                                                    "std::string constexpr",
                                                    common_text);
  const size_t view_index = add_arg_handle_sample(samples,
                                                  {ArgType::NON_CONSTEXPR},
                                                  fmt::format("{}", common_view),
                                                  "std::string_view routed as constexpr",
                                                  common_view);
  const size_t c_string_index = add_arg_handle_sample(samples,
                                                      {ArgType::SPECIALIZED},
                                                      fmt::format("{}", common_pointer),
                                                      "C string routed as constexpr",
                                                      common_pointer);
  ok &= expect(samples[string_index].key == samples[view_index].key &&
                   samples[string_index].key == samples[c_string_index].key,
               "equivalent string representations must share a structural key");

  const std::string embedded_null("left\0right", 10);
  const std::string_view embedded_null_view(embedded_null.data(), embedded_null.size());
  const size_t embedded_string_index = add_arg_handle_sample(samples,
                                                             {ArgType::CONSTEXPR},
                                                             fmt::format("{}", embedded_null),
                                                             "string with embedded NUL",
                                                             embedded_null);
  const size_t embedded_view_index = add_arg_handle_sample(samples,
                                                           {ArgType::CONSTEXPR},
                                                           fmt::format("{}", embedded_null_view),
                                                           "string_view with embedded NUL",
                                                           embedded_null_view);
  ok &= expect(samples[embedded_string_index].key == samples[embedded_view_index].key,
               "equal string and string_view byte sequences must share a structural key");
  add_arg_handle_sample(samples,
                        {ArgType::CONSTEXPR},
                        "nullopt",
                        "text intentionally colliding with the legacy nullopt spelling",
                        std::string("nullopt"));
  add_arg_handle_sample(samples,
                        {ArgType::CONSTEXPR},
                        "true",
                        "text intentionally colliding with the legacy bool spelling",
                        std::string("true"));

  const auto composite_tuple = std::tuple<float, int> {2.5F, 7};
  const TritonDevicePtr composite_pointer {pointer_values[2], TritonDType::kFp32};
  const std::string composite_text = "tl.float16";
  const double composite_float = -4.75;
  const std::string composite_legacy = legacy_pointer_signature(ArgType::SPECIALIZED, composite_pointer) +
                                       "," + legacy_tuple_signature(composite_tuple) + ",nullopt," +
                                       composite_text + "," +
                                       legacy_scalar_signature(ArgType::CONSTEXPR, composite_float);
  add_arg_handle_sample(samples,
                        {ArgType::SPECIALIZED,
                         ArgType::NON_CONSTEXPR,
                         ArgType::SPECIALIZED,
                         ArgType::CONSTEXPR,
                         ArgType::CONSTEXPR},
                        composite_legacy,
                        "mixed production argument list",
                        composite_pointer,
                        composite_tuple,
                        std::optional<int64_t> {},
                        composite_text,
                        composite_float);

  for (const auto& sample : samples) {
    ok &= expect(render_signature(sample.key) == sample.legacy_signature,
                 fmt::format("ArgHandle rendering must match the legacy path for {}", sample.label));
  }
  for (size_t i = 0; i < samples.size(); ++i) {
    for (size_t j = i; j < samples.size(); ++j) {
      if (samples[i].key == samples[j].key) {
        ok &= expect(samples[i].legacy_signature == samples[j].legacy_signature,
                     fmt::format("equal ArgHandle keys must not merge legacy signatures: {} vs {}",
                                 samples[i].label,
                                 samples[j].label));
        ok &= expect(samples[i].key.hash() == samples[j].key.hash(),
                     "equal ArgHandle keys must have equal hashes");
      }
    }
  }
  return ok;
}

bool test_arg_handle_runtime_abi() {
#if defined(BACKEND_NPU)
  constexpr size_t kSpecializedOneArgs = 1;
#else
  constexpr size_t kSpecializedOneArgs = 0;
#endif

  bool ok = true;
  ok &= expect(emit_arg_handle_abi_count({ArgType::NON_CONSTEXPR}, int64_t {1}) == 1,
               "a non-constexpr scalar must remain in the runtime ABI");
  ok &= expect(emit_arg_handle_abi_count({ArgType::SPECIALIZED}, int64_t {1}) == kSpecializedOneArgs,
               "equal-to-one specialization must preserve the legacy runtime ABI");
  ok &= expect(emit_arg_handle_abi_count({ArgType::SPECIALIZED}, int64_t {16}) == 1,
               "alignment specialization must keep its scalar runtime argument");
  ok &= expect(emit_arg_handle_abi_count({ArgType::CONSTEXPR}, int64_t {16}) == 0,
               "a constexpr scalar must not enter the runtime ABI");
  ok &= expect(emit_arg_handle_abi_count({ArgType::NON_CONSTEXPR}, nullptr) == 1,
               "a runtime nullptr must occupy one pointer ABI slot");

  const auto tuple =
      std::tuple<int, unsigned int, int64_t, uint64_t, float, double> {-3, 4U, -5, 6U, 7.5F, 8.5};
  ok &= expect(emit_arg_handle_abi_count({ArgType::NON_CONSTEXPR}, tuple) == 6,
               "a runtime tuple must preserve one ABI slot per element");
  ok &= expect(emit_arg_handle_abi_count({ArgType::SPECIALIZED}, std::optional<int64_t> {}) == 0,
               "an empty optional must not enter the runtime ABI");
  ok &= expect(emit_arg_handle_abi_count({ArgType::SPECIALIZED}, std::optional<int64_t> {16}) == 1,
               "an engaged optional must preserve its value's runtime ABI");
  return ok;
}

bool test_legacy_arg_handle_api() {
  static_assert(std::is_aggregate_v<triton_jit::ArgHandle>);
  static_assert(std::is_same_v<decltype(triton_jit::ArgHandle::signature), c10::SmallVector<std::string>&>);

  StaticSignature static_signature {
      3,
      {ArgType::SPECIALIZED, ArgType::CONSTEXPR, ArgType::NON_CONSTEXPR}
  };
  ParameterBuffer buffer;
  c10::SmallVector<std::string> signature;
  triton_jit::ArgHandle handler {static_signature, buffer, signature, 0};
  handler.handle_args(int64_t {16}, std::string("tl.float32"), nullptr);

#if defined(BACKEND_NPU)
  constexpr std::string_view kIntegerSignature = "i32";
#else
  constexpr std::string_view kIntegerSignature = "i32:16";
#endif
  bool ok = expect(handler.idx == 3, "legacy ArgHandle must consume every signature slot");
  ok &= expect(buffer.size() == 2, "legacy ArgHandle must preserve runtime ABI entries");
  ok &= expect(signature.size() == 3 && signature[0] == kIntegerSignature && signature[1] == "tl.float32" &&
                   signature[2] == "*i8",
               "legacy ArgHandle must preserve its public string-token output");

  for (const auto invoke : {
           +[](triton_jit::ArgHandle& direct) { direct.handle_specialized(nullptr); },
           +[](triton_jit::ArgHandle& direct) { direct.handle_specialized_no_alignment(nullptr); },
           +[](triton_jit::ArgHandle& direct) { direct.handle_non_constexpr(nullptr); },
       }) {
    StaticSignature direct_signature {1, {ArgType::NON_CONSTEXPR}};
    ParameterBuffer direct_buffer;
    c10::SmallVector<std::string> direct_tokens;
    triton_jit::ArgHandle direct {direct_signature, direct_buffer, direct_tokens, 0};
    invoke(direct);
    ok &= expect(direct_buffer.size() == 1 && direct_tokens.size() == 1 && direct_tokens[0] == "*i8",
                 "direct legacy nullptr handlers must preserve the *i8 ABI token");
  }
  return ok;
}

bool test_invalid_atoms() {
  bool ok = true;
  ok &= expect_throws<std::invalid_argument>(
      [] {
        SignatureKey key;
        key.append_runtime_pointer(TritonDType::kFp32, SignatureSpecialization::kEqualToOne);
      },
      "pointers must reject equal-to-one specialization");
  ok &= expect_throws<std::invalid_argument>(
      [] {
        SignatureKey key;
        key.append_runtime_scalar(TritonDType::kFp32, SignatureSpecialization::kDivisibleBy16);
      },
      "floating runtime scalars must reject integer specialization");
  ok &= expect_throws<std::invalid_argument>(
      [] {
        SignatureKey key;
        key.append_runtime_tuple({});
      },
      "runtime tuples must not be empty");
  ok &= expect_throws<std::invalid_argument>(
      [] {
        SignatureKey key;
        key.append_runtime_scalar(static_cast<TritonDType>(255));
      },
      "invalid dtype values must be rejected");
  ok &= expect_throws<std::invalid_argument>(
      [] {
        SignatureKey key;
        key.append_runtime_scalar(TritonDType::kI32, static_cast<SignatureSpecialization>(255));
      },
      "invalid specialization values must be rejected");
  ok &= expect_throws<std::logic_error>(
      [] {
        SignatureKey key = SignatureKey::raw_fallback("i32");
        key.append_nullopt();
      },
      "raw fallback signatures must reject additional atoms");
  return ok;
}

SignatureKey make_runtime_signature(TritonDType dtype = TritonDType::kFp32) {
  SignatureKey signature;
  signature.append_runtime_pointer(dtype, SignatureSpecialization::kDivisibleBy16);
  signature.append_runtime_scalar(TritonDType::kI32);
  return signature;
}

bool test_kernel_cache_key() {
  bool ok = true;

  CompileOptions options;
  options.num_warps = 8;
  options.num_stages = 4;
  KernelCacheKey baseline = make_kernel_cache_key(make_runtime_signature(), 2, options);
  KernelCacheKey equal = make_kernel_cache_key(make_runtime_signature(), 2, options);
  ok &= expect(baseline == equal, "equal kernel inputs must produce equal keys");
  ok &= expect(baseline.hash() == equal.hash(), "equal kernel keys must share a hash");
  ok &= expect(KernelCacheKeyHash {}(baseline) == baseline.hash(),
               "unordered cache hasher must return the cached hash");

  ok &= expect(!(baseline == make_kernel_cache_key(make_runtime_signature(), 3, options)),
               "device index must participate in the kernel key");

  CompileOptions different_warps = options;
  different_warps.num_warps = 4;
  ok &= expect(!(baseline == make_kernel_cache_key(make_runtime_signature(), 2, different_warps)),
               "num_warps must participate in the kernel key");

  CompileOptions different_stages = options;
  different_stages.num_stages = 5;
  ok &= expect(!(baseline == make_kernel_cache_key(make_runtime_signature(), 2, different_stages)),
               "num_stages must participate in the kernel key");

  ok &= expect(!(baseline == make_kernel_cache_key(make_runtime_signature(TritonDType::kFp16), 2, options)),
               "the structural signature must participate in the kernel key");

  CompileOptions with_extra = options;
  with_extra.extra.emplace("debug", "true");
  with_extra.extra.emplace("opt_level", "O2");
  KernelCacheKey owned_extra = make_kernel_cache_key(make_runtime_signature(), 2, with_extra);
  with_extra.extra["opt_level"] = "O3";
  with_extra.extra.emplace("new_option", "new_value");
  const std::map<std::string, std::string> expected_extra {
      {    "debug", "true"},
      {"opt_level",   "O2"}
  };
  ok &= expect(owned_extra.extra() == expected_extra,
               "kernel key must own an exact snapshot of CompileOptions.extra");

  std::map<std::string, std::string> reverse_inserted;
  reverse_inserted.emplace("opt_level", "O2");
  reverse_inserted.emplace("debug", "true");
  KernelCacheKey same_extra {make_runtime_signature(),
                             2,
                             options.num_warps,
                             options.num_stages,
                             reverse_inserted};
  ok &=
      expect(owned_extra == same_extra, "ordered maps with identical entries must produce equal kernel keys");

  CompileOptions changed_extra = options;
  changed_extra.extra.emplace("debug", "false");
  changed_extra.extra.emplace("opt_level", "O2");
  ok &= expect(!(owned_extra == make_kernel_cache_key(make_runtime_signature(), 2, changed_extra)),
               "extra option values must participate in full equality");

  CompileOptions invalid_options = options;
  invalid_options.extra.emplace("num_warps", "16");
  ok &= expect_throws<std::invalid_argument>(
      [&] { (void)make_kernel_cache_key(make_runtime_signature(), 2, invalid_options); },
      "reserved compile options must remain rejected");
  return ok;
}

bool test_empty_extra_allocation() {
  CompileOptions options;
  SignatureKey signature = make_runtime_signature();

  const size_t before = tracked_allocations.load(std::memory_order_relaxed);
  track_allocations = true;
  KernelCacheKey key = make_kernel_cache_key(std::move(signature), 0, options);
  track_allocations = false;
  const size_t after = tracked_allocations.load(std::memory_order_relaxed);

  bool ok =
      expect(after == before, "an empty CompileOptions.extra map must not allocate while building a key");
  ok &= expect(key.extra().empty(), "an empty extra map must remain empty in the key");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_signature_partition_and_rendering();
  ok &= test_arg_handle_legacy_properties();
  ok &= test_arg_handle_runtime_abi();
  ok &= test_legacy_arg_handle_api();
  ok &= test_invalid_atoms();
  ok &= test_kernel_cache_key();
  ok &= test_empty_extra_allocation();
  return ok ? 0 : 1;
}
