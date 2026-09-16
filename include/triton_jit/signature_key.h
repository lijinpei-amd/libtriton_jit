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

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include "c10/util/SmallVector.h"
#include "triton_jit/device_ptr.h"

namespace triton_jit {

struct CompileOptions;

namespace detail {

  enum class SignatureAtomKind : uint8_t {
    kRuntimeScalar = 1,
    kRuntimePointer,
    kRuntimeTuple,
    kConstexprInteger,
    kConstexprBool,
    kConstexprF32,
    kConstexprF64,
    kConstexprString,
    kNullopt,
    kJitFunction,
    kRuntimeText,
    kRawFallback,
  };

  enum class SignatureSpecialization : uint8_t {
    kNone = 0,
    kDivisibleBy16,
    kEqualToOne,
  };

  // An owned structural representation of a Triton signature. Common runtime
  // arguments need one uint64_t each and stay inside the 16-word inline buffer.
  // Exact text is retained only for constexpr strings, JITFunction tokens, and
  // raw-signature fallbacks, so equality never relies on a hash alone.
  class SignatureKey {
   public:
    SignatureKey() = default;

    void append_runtime_scalar(TritonDType dtype,
                               SignatureSpecialization specialization = SignatureSpecialization::kNone);
    void append_runtime_pointer(TritonDType dtype,
                                SignatureSpecialization specialization = SignatureSpecialization::kNone);
    void append_runtime_tuple(std::span<const TritonDType> element_types);

    template <std::integral T>
      requires(!std::same_as<std::remove_cv_t<T>, bool>)
    void append_constexpr_integer(T value) {
      static_assert(sizeof(T) <= sizeof(uint64_t),
                    "integer signature atoms wider than 64 bits are unsupported");
      using Unsigned = std::make_unsigned_t<T>;
      if constexpr (std::is_signed_v<T>) {
        if (value < 0) {
          const Unsigned unsigned_value = static_cast<Unsigned>(value);
          const Unsigned magnitude = static_cast<Unsigned>(Unsigned {0} - unsigned_value);
          append_constexpr_integer_parts(true, static_cast<uint64_t>(magnitude));
          return;
        }
      }
      append_constexpr_integer_parts(false, static_cast<uint64_t>(value));
    }

    void append_constexpr_bool(bool value);
    void append_constexpr_f32(float value);
    void append_constexpr_f64(double value);
    void append_constexpr_string(std::string_view value);
    void append_nullopt();
    void append_jit_function(std::string_view exact_token);
    void append_runtime_text(std::string_view exact_token);

    static SignatureKey raw_fallback(std::string_view exact_signature);

    size_t size() const noexcept {
      return atom_count_;
    }

    bool empty() const noexcept {
      return atom_count_ == 0;
    }

    bool is_raw_fallback() const noexcept {
      return raw_fallback_;
    }

    size_t hash() const noexcept {
      return static_cast<size_t>(hash_);
    }

    friend bool operator==(const SignatureKey& lhs, const SignatureKey& rhs) noexcept {
      return lhs.hash_ == rhs.hash_ && lhs.atom_count_ == rhs.atom_count_ &&
             lhs.raw_fallback_ == rhs.raw_fallback_ && lhs.words_ == rhs.words_ &&
             lhs.payloads_ == rhs.payloads_;
    }

   private:
    friend std::string render_signature(const SignatureKey& key);

    static constexpr uint64_t kHashSeed = 14695981039346656037ULL;

    void ensure_structural_append() const;
    void append_constexpr_integer_parts(bool negative, uint64_t magnitude);
    void append_text_atom(SignatureAtomKind kind, std::string_view value);
    void append_word(uint64_t word);
    uint32_t append_payload(std::string_view value);
    void hash_bytes(std::string_view value);

    c10::SmallVector<uint64_t, 16> words_;
    c10::SmallVector<std::string, 1> payloads_;
    uint64_t hash_ = kHashSeed;
    size_t atom_count_ = 0;
    bool raw_fallback_ = false;
  };

  std::string render_signature(const SignatureKey& key);

  // The complete kernel-cache identity. CompileOptions remains source-compatible:
  // this key snapshots its public map, and the default empty std::map allocates no
  // nodes. The cached hash accelerates lookup; equality still compares every field.
  class KernelCacheKey {
   public:
    KernelCacheKey(SignatureKey signature,
                   int device_index,
                   int num_warps,
                   int num_stages,
                   std::map<std::string, std::string> extra = {});
    KernelCacheKey(SignatureKey signature, int device_index, const CompileOptions& options);

    const SignatureKey& signature() const noexcept {
      return signature_;
    }

    int device_index() const noexcept {
      return device_index_;
    }

    int num_warps() const noexcept {
      return num_warps_;
    }

    int num_stages() const noexcept {
      return num_stages_;
    }

    const std::map<std::string, std::string>& extra() const noexcept {
      return extra_;
    }

    size_t hash() const noexcept {
      return static_cast<size_t>(hash_);
    }

    friend bool operator==(const KernelCacheKey& lhs, const KernelCacheKey& rhs) noexcept {
      return lhs.hash_ == rhs.hash_ && lhs.device_index_ == rhs.device_index_ &&
             lhs.num_warps_ == rhs.num_warps_ && lhs.num_stages_ == rhs.num_stages_ &&
             lhs.signature_ == rhs.signature_ && lhs.extra_ == rhs.extra_;
    }

   private:
    SignatureKey signature_;
    int device_index_;
    int num_warps_;
    int num_stages_;
    std::map<std::string, std::string> extra_;
    uint64_t hash_;
  };

  struct KernelCacheKeyHash {
    size_t operator()(const KernelCacheKey& key) const noexcept {
      return key.hash();
    }
  };

  KernelCacheKey make_kernel_cache_key(SignatureKey signature,
                                       int device_index,
                                       const CompileOptions& options);

}  // namespace detail
}  // namespace triton_jit
