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

#include "triton_jit/signature_key.h"

#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

#include "fmt/core.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit::detail {
namespace {

  constexpr uint64_t kHashPrime = 1099511628211ULL;

  constexpr uint64_t make_header(SignatureAtomKind kind,
                                 TritonDType dtype = TritonDType::kI1,
                                 SignatureSpecialization specialization = SignatureSpecialization::kNone,
                                 uint32_t auxiliary = 0) {
    return static_cast<uint64_t>(kind) | (static_cast<uint64_t>(dtype) << 8) |
           (static_cast<uint64_t>(specialization) << 16) | (static_cast<uint64_t>(auxiliary) << 32);
  }

  SignatureAtomKind atom_kind(uint64_t header) {
    return static_cast<SignatureAtomKind>(header & 0xffU);
  }

  TritonDType atom_dtype(uint64_t header) {
    return static_cast<TritonDType>((header >> 8) & 0xffU);
  }

  SignatureSpecialization atom_specialization(uint64_t header) {
    return static_cast<SignatureSpecialization>((header >> 16) & 0xffU);
  }

  uint32_t atom_auxiliary(uint64_t header) {
    return static_cast<uint32_t>(header >> 32);
  }

  bool is_valid_dtype(TritonDType dtype) {
    switch (dtype) {
      case TritonDType::kI1:
      case TritonDType::kI8:
      case TritonDType::kI16:
      case TritonDType::kI32:
      case TritonDType::kI64:
      case TritonDType::kU8:
      case TritonDType::kU16:
      case TritonDType::kU32:
      case TritonDType::kU64:
      case TritonDType::kFp16:
      case TritonDType::kBf16:
      case TritonDType::kFp32:
      case TritonDType::kFp64:
      case TritonDType::kFp8E4NV:
      case TritonDType::kFp8E5:
        return true;
    }
    return false;
  }

  bool is_integer_dtype(TritonDType dtype) {
    switch (dtype) {
      case TritonDType::kI1:
      case TritonDType::kI8:
      case TritonDType::kI16:
      case TritonDType::kI32:
      case TritonDType::kI64:
      case TritonDType::kU8:
      case TritonDType::kU16:
      case TritonDType::kU32:
      case TritonDType::kU64:
        return true;
      case TritonDType::kFp16:
      case TritonDType::kBf16:
      case TritonDType::kFp32:
      case TritonDType::kFp64:
      case TritonDType::kFp8E4NV:
      case TritonDType::kFp8E5:
        return false;
    }
    return false;
  }

  void validate_dtype(TritonDType dtype) {
    if (!is_valid_dtype(dtype)) {
      throw std::invalid_argument("invalid Triton dtype in signature key");
    }
  }

  void validate_specialization(SignatureSpecialization specialization) {
    switch (specialization) {
      case SignatureSpecialization::kNone:
      case SignatureSpecialization::kDivisibleBy16:
      case SignatureSpecialization::kEqualToOne:
        return;
    }
    throw std::invalid_argument("invalid specialization in signature key");
  }

  std::string_view specialization_suffix(SignatureSpecialization specialization) {
    switch (specialization) {
      case SignatureSpecialization::kNone:
        return {};
      case SignatureSpecialization::kDivisibleBy16:
        return ":16";
      case SignatureSpecialization::kEqualToOne:
        return ":1";
    }
    throw std::logic_error("invalid signature specialization");
  }

  void append_hash_word(uint64_t& hash, uint64_t word) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      hash ^= static_cast<uint8_t>(word >> shift);
      hash *= kHashPrime;
    }
  }

  void append_hash_bytes(uint64_t& hash, std::string_view bytes) {
    append_hash_word(hash, bytes.size());
    for (const unsigned char byte : bytes) {
      hash ^= byte;
      hash *= kHashPrime;
    }
  }

  uint64_t make_kernel_hash(const SignatureKey& signature,
                            int device_index,
                            int num_warps,
                            int num_stages,
                            const std::map<std::string, std::string>& extra) {
    uint64_t hash = 14695981039346656037ULL;
    append_hash_word(hash, signature.hash());
    append_hash_word(hash, static_cast<uint32_t>(device_index));
    append_hash_word(hash, static_cast<uint32_t>(num_warps));
    append_hash_word(hash, static_cast<uint32_t>(num_stages));
    append_hash_word(hash, extra.size());
    for (const auto& [name, value] : extra) {
      append_hash_bytes(hash, name);
      append_hash_bytes(hash, value);
    }
    return hash;
  }

}  // namespace

void SignatureKey::ensure_structural_append() const {
  if (raw_fallback_) {
    throw std::logic_error("cannot append to a raw signature fallback");
  }
}

void SignatureKey::append_word(uint64_t word) {
  words_.push_back(word);
  append_hash_word(hash_, word);
}

uint32_t SignatureKey::append_payload(std::string_view value) {
  if (payloads_.size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("too many signature payloads");
  }
  const uint32_t index = static_cast<uint32_t>(payloads_.size());
  payloads_.emplace_back(value);
  hash_bytes(value);
  return index;
}

void SignatureKey::hash_bytes(std::string_view value) {
  append_hash_bytes(hash_, value);
}

void SignatureKey::append_runtime_scalar(TritonDType dtype, SignatureSpecialization specialization) {
  ensure_structural_append();
  validate_dtype(dtype);
  validate_specialization(specialization);
  if (specialization != SignatureSpecialization::kNone && !is_integer_dtype(dtype)) {
    throw std::invalid_argument("only integer runtime scalars may be specialized");
  }
  append_word(make_header(SignatureAtomKind::kRuntimeScalar, dtype, specialization));
  ++atom_count_;
}

void SignatureKey::append_runtime_pointer(TritonDType dtype, SignatureSpecialization specialization) {
  ensure_structural_append();
  validate_dtype(dtype);
  validate_specialization(specialization);
  if (specialization == SignatureSpecialization::kEqualToOne) {
    throw std::invalid_argument("runtime pointers cannot use equal-to-one specialization");
  }
  append_word(make_header(SignatureAtomKind::kRuntimePointer, dtype, specialization));
  ++atom_count_;
}

void SignatureKey::append_runtime_tuple(std::span<const TritonDType> element_types) {
  ensure_structural_append();
  if (element_types.empty()) {
    throw std::invalid_argument("runtime tuple signature atoms must not be empty");
  }
  if (element_types.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("runtime tuple signature atom is too large");
  }
  for (const TritonDType dtype : element_types) {
    validate_dtype(dtype);
  }
  append_word(make_header(SignatureAtomKind::kRuntimeTuple,
                          TritonDType::kI1,
                          SignatureSpecialization::kNone,
                          static_cast<uint32_t>(element_types.size())));
  for (const TritonDType dtype : element_types) {
    append_word(static_cast<uint64_t>(dtype));
  }
  ++atom_count_;
}

void SignatureKey::append_constexpr_integer_parts(bool negative, uint64_t magnitude) {
  ensure_structural_append();
  negative = negative && magnitude != 0;
  append_word(make_header(SignatureAtomKind::kConstexprInteger,
                          TritonDType::kI1,
                          SignatureSpecialization::kNone,
                          negative ? 1U : 0U));
  append_word(magnitude);
  ++atom_count_;
}

void SignatureKey::append_constexpr_bool(bool value) {
  ensure_structural_append();
  append_word(make_header(SignatureAtomKind::kConstexprBool,
                          TritonDType::kI1,
                          SignatureSpecialization::kNone,
                          value ? 1U : 0U));
  ++atom_count_;
}

void SignatureKey::append_constexpr_f32(float value) {
  ensure_structural_append();
  append_word(make_header(SignatureAtomKind::kConstexprF32));
  append_word(std::bit_cast<uint32_t>(value));
  ++atom_count_;
}

void SignatureKey::append_constexpr_f64(double value) {
  ensure_structural_append();
  append_word(make_header(SignatureAtomKind::kConstexprF64));
  append_word(std::bit_cast<uint64_t>(value));
  ++atom_count_;
}

void SignatureKey::append_text_atom(SignatureAtomKind kind, std::string_view value) {
  ensure_structural_append();
  if (payloads_.size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("too many signature payloads");
  }
  const uint32_t payload_index = static_cast<uint32_t>(payloads_.size());
  append_word(make_header(kind, TritonDType::kI1, SignatureSpecialization::kNone, payload_index));
  (void)append_payload(value);
  ++atom_count_;
}

void SignatureKey::append_constexpr_string(std::string_view value) {
  append_text_atom(SignatureAtomKind::kConstexprString, value);
}

void SignatureKey::append_nullopt() {
  ensure_structural_append();
  append_word(make_header(SignatureAtomKind::kNullopt));
  ++atom_count_;
}

void SignatureKey::append_jit_function(std::string_view exact_token) {
  append_text_atom(SignatureAtomKind::kJitFunction, exact_token);
}

void SignatureKey::append_runtime_text(std::string_view exact_token) {
  append_text_atom(SignatureAtomKind::kRuntimeText, exact_token);
}

SignatureKey SignatureKey::raw_fallback(std::string_view exact_signature) {
  SignatureKey key;
  key.raw_fallback_ = true;
  const uint32_t payload_index = key.append_payload(exact_signature);
  key.append_word(make_header(SignatureAtomKind::kRawFallback,
                              TritonDType::kI1,
                              SignatureSpecialization::kNone,
                              payload_index));
  key.atom_count_ = 1;
  return key;
}

std::string render_signature(const SignatureKey& key) {
  std::string signature;
  bool first_atom = true;
  size_t word_index = 0;

  auto append_separator = [&]() {
    if (!first_atom) {
      signature.push_back(',');
    }
    first_atom = false;
  };

  auto payload_at = [&](uint32_t index) -> const std::string& {
    if (index >= key.payloads_.size()) {
      throw std::logic_error("corrupt signature payload index");
    }
    return key.payloads_[index];
  };

  while (word_index < key.words_.size()) {
    const uint64_t header = key.words_[word_index++];
    const SignatureAtomKind kind = atom_kind(header);
    if (kind == SignatureAtomKind::kRawFallback) {
      if (key.atom_count_ != 1 || word_index != 1) {
        throw std::logic_error("raw signature fallback must be the only atom");
      }
      return payload_at(atom_auxiliary(header));
    }

    append_separator();
    switch (kind) {
      case SignatureAtomKind::kRuntimeScalar:
        signature += to_triton_typename(atom_dtype(header));
        signature += specialization_suffix(atom_specialization(header));
        break;
      case SignatureAtomKind::kRuntimePointer:
        signature.push_back('*');
        signature += to_triton_typename(atom_dtype(header));
        signature += specialization_suffix(atom_specialization(header));
        break;
      case SignatureAtomKind::kRuntimeTuple: {
        const uint32_t element_count = atom_auxiliary(header);
        if (word_index + element_count > key.words_.size()) {
          throw std::logic_error("corrupt runtime tuple signature atom");
        }
        signature.push_back('(');
        for (uint32_t i = 0; i < element_count; ++i) {
          if (i != 0) {
            signature.push_back(',');
          }
          signature += to_triton_typename(static_cast<TritonDType>(key.words_[word_index++]));
        }
        signature.push_back(')');
        break;
      }
      case SignatureAtomKind::kConstexprInteger: {
        if (word_index == key.words_.size()) {
          throw std::logic_error("corrupt integer constexpr signature atom");
        }
        const uint64_t magnitude = key.words_[word_index++];
        if (atom_auxiliary(header) != 0) {
          signature.push_back('-');
        }
        signature += std::to_string(magnitude);
        break;
      }
      case SignatureAtomKind::kConstexprBool:
        signature += atom_auxiliary(header) == 0 ? "false" : "true";
        break;
      case SignatureAtomKind::kConstexprF32: {
        if (word_index == key.words_.size()) {
          throw std::logic_error("corrupt fp32 constexpr signature atom");
        }
        const float value = std::bit_cast<float>(static_cast<uint32_t>(key.words_[word_index++]));
        signature += fmt::format("{}", value);
        break;
      }
      case SignatureAtomKind::kConstexprF64: {
        if (word_index == key.words_.size()) {
          throw std::logic_error("corrupt fp64 constexpr signature atom");
        }
        const double value = std::bit_cast<double>(key.words_[word_index++]);
        signature += fmt::format("{}", value);
        break;
      }
      case SignatureAtomKind::kConstexprString:
      case SignatureAtomKind::kJitFunction:
      case SignatureAtomKind::kRuntimeText:
        signature += payload_at(atom_auxiliary(header));
        break;
      case SignatureAtomKind::kNullopt:
        signature += "nullopt";
        break;
      case SignatureAtomKind::kRawFallback:
        throw std::logic_error("unreachable raw signature fallback");
    }
  }

  return signature;
}

KernelCacheKey::KernelCacheKey(SignatureKey signature,
                               int device_index,
                               int num_warps,
                               int num_stages,
                               std::map<std::string, std::string> extra)
    : signature_(std::move(signature)),
      device_index_(device_index),
      num_warps_(num_warps),
      num_stages_(num_stages),
      extra_(std::move(extra)),
      hash_(make_kernel_hash(signature_, device_index_, num_warps_, num_stages_, extra_)) {
}

KernelCacheKey::KernelCacheKey(SignatureKey signature, int device_index, const CompileOptions& options)
    : KernelCacheKey(
          std::move(signature), device_index, options.num_warps, options.num_stages, options.extra) {
  validate_compile_options(options);
}

KernelCacheKey make_kernel_cache_key(SignatureKey signature,
                                     int device_index,
                                     const CompileOptions& options) {
  return KernelCacheKey(std::move(signature), device_index, options);
}

}  // namespace triton_jit::detail
