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

#include <torch/cuda.h>
#include <torch/torch.h>

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "triton_jit/backend_config.h"
#include "triton_jit/device_ptr.h"
#include "triton_jit/triton_jit_function.h"

namespace {

constexpr int kWarmupIterations = 50;
constexpr int kBatchIterations = 5000;
constexpr int kSamples = 7;
constexpr unsigned int kNumWarps = 1;
constexpr unsigned int kNumStages = 1;
constexpr int kKernelColumnWidth = 28;
constexpr int kLevelColumnWidth = 24;

volatile size_t benchmark_sink = 0;

template <size_t... Indices>
auto make_argument_tuple(triton_jit::TritonDevicePtr output,
                         triton_jit::TritonDevicePtr input,
                         std::index_sequence<Indices...>) {
  return std::tuple {((void)Indices, Indices == 0 ? output : input)...};
}

template <size_t NumArgs>
auto make_argument_tuple(triton_jit::TritonDevicePtr output,
                         triton_jit::TritonDevicePtr input) {
  return make_argument_tuple(output, input, std::make_index_sequence<NumArgs> {});
}

template <size_t... Indices>
auto make_constexpr_heavy_argument_tuple(triton_jit::TritonDevicePtr output,
                                         std::index_sequence<Indices...>) {
  return std::tuple {output, static_cast<int32_t>(Indices + 1)...};
}

auto make_constexpr_heavy_argument_tuple(triton_jit::TritonDevicePtr output) {
  return make_constexpr_heavy_argument_tuple(output, std::make_index_sequence<29> {});
}

template <size_t... PointerIndices, size_t... ConstexprIndices>
auto make_mixed_argument_tuple(triton_jit::TritonDevicePtr output,
                               triton_jit::TritonDevicePtr input,
                               std::index_sequence<PointerIndices...>,
                               std::index_sequence<ConstexprIndices...>) {
  return std::tuple_cat(
      std::tuple {output},
      std::tuple {((void)PointerIndices, input)...},
      std::tuple {static_cast<int32_t>(ConstexprIndices + 1)...});
}

auto make_mixed_argument_tuple(triton_jit::TritonDevicePtr output,
                               triton_jit::TritonDevicePtr input) {
  return make_mixed_argument_tuple(
      output, input, std::make_index_sequence<14> {}, std::make_index_sequence<15> {});
}

struct PreparedArguments {
  triton_jit::ParameterBuffer buffer;
  triton_jit::detail::SignatureKey signature;
  std::string full_signature;
};

template <typename Tuple>
PreparedArguments prepare_arguments(const triton_jit::StaticSignature& static_signature,
                                    const Tuple& arguments,
                                    bool render_signature_text = true) {
  PreparedArguments prepared;
  prepared.buffer.reserve(static_signature.num_args + 2);
  triton_jit::detail::StructuralArgHandle handler {static_signature, prepared.buffer, prepared.signature, 0};
  std::apply([&](const auto&... args) { (handler.handle_arg(args), ...); }, arguments);

#if !defined(BACKEND_NPU)
  handler.append_global_scratch();
  handler.append_global_scratch();
#endif

  if (render_signature_text) {
    prepared.full_signature = triton_jit::detail::render_signature(prepared.signature);
  }
  return prepared;
}

template <typename Backend>
unsigned int get_warp_size(const triton_jit::TritonKernelImpl<Backend>& kernel) {
  if constexpr (triton_jit::DynamicWarpSizeBackend<Backend>) {
    return Backend::get_warp_size(kernel.get_dir(), kernel.get_kernel_name());
  } else {
    return Backend::WARP_SIZE;
  }
}

void synchronize_device() {
  torch::cuda::synchronize();
}

template <typename Function>
double measure_us(Function&& function) {
  for (int i = 0; i < kWarmupIterations; ++i) {
    function();
  }
  synchronize_device();

  std::array<double, kSamples> samples {};
  for (double& sample : samples) {
    synchronize_device();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kBatchIterations; ++i) {
      function();
    }
    const auto end = std::chrono::steady_clock::now();
    synchronize_device();
    sample = std::chrono::duration<double, std::micro>(end - start).count() /
             static_cast<double>(kBatchIterations);
  }

  std::sort(samples.begin(), samples.end());
  return samples[kSamples / 2];
}

void print_result(const std::string& kernel,
                  const std::string& level,
                  double microseconds) {
  const std::string name = kernel + "/" + level;
  std::cout << std::left << std::setw(kKernelColumnWidth) << kernel
            << std::setw(kLevelColumnWidth) << level << std::right
            << std::fixed << std::setprecision(3) << std::setw(10) << microseconds << '\n';
  std::cout << "BENCH " << name << " " << std::fixed << std::setprecision(6) << microseconds
            << '\n';
}

size_t count_signature_suffix(std::string_view signature, std::string_view suffix) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = signature.find(suffix, offset)) != std::string_view::npos) {
    const size_t token_end = offset + suffix.size();
    if (token_end == signature.size() || signature[token_end] == ',' ||
        signature[token_end] == ')') {
      ++count;
    }
    offset = token_end;
  }
  return count;
}

void print_case_profile(const std::string& benchmark_name,
                        const triton_jit::StaticSignature& signature,
                        std::string_view rendered_signature,
                        size_t launch_parameters) {
  size_t constexpr_count = 0;
  size_t specialized_count = 0;
  size_t specialized_no_alignment_count = 0;
  for (triton_jit::ArgType type : signature.arg_type) {
    constexpr_count += type == triton_jit::ArgType::CONSTEXPR;
    specialized_count += type == triton_jit::ArgType::SPECIALIZED;
    specialized_no_alignment_count += type == triton_jit::ArgType::SPECIALIZED_NO_ALIGNMENT;
  }

  const size_t div16_count = count_signature_suffix(rendered_signature, ":16");
  const size_t equal1_count = count_signature_suffix(rendered_signature, ":1");

  std::cout << "CASE " << benchmark_name << " declared=" << signature.num_args
            << " dynamic=" << static_cast<size_t>(signature.num_args) - constexpr_count
            << " constexpr=" << constexpr_count << " specialized=" << specialized_count
            << " specialized_no_alignment=" << specialized_no_alignment_count
            << " div16=" << div16_count << " equal1=" << equal1_count
            << " launch_parameters=" << launch_parameters << '\n';
}

template <typename Function>
void validate_launch(const std::string& kernel,
                     const std::string& level,
                     at::Tensor& output,
                     float expected,
                     Function&& function) {
  output.fill_(-1.0F);
  synchronize_device();
  function();
  synchronize_device();

  const float actual = output.item<float>();
  if (actual != expected) {
    throw std::runtime_error(kernel + "/" + level + " produced " +
                             std::to_string(actual) + ", expected " +
                             std::to_string(expected));
  }
}

template <typename Tuple>
void run_kernel_benchmark(const std::filesystem::path& fixture,
                          const std::string& benchmark_name,
                          const std::string& function_name,
                          at::Tensor& output_tensor,
                          Tuple arguments,
                          float expected) {
  using Backend = triton_jit::DefaultBackend;

  auto& function = triton_jit::TritonJITFunction::get_instance(fixture.string(), function_name);
  typename Backend::StreamType stream {};
  triton_jit::CompileOptions options;
  options.num_warps = kNumWarps;
  options.num_stages = kNumStages;

  auto full_launch = [&]() {
    std::apply(
        [&](const auto&... args) {
          function(stream, 1, 1, 1, options, args...);
        },
        arguments);
  };

  // Compile the kernel, populate all caches, and verify the typed launch path.
  validate_launch(benchmark_name,
                  "operator",
                  output_tensor,
                  expected,
                  full_launch);

  PreparedArguments prepared = prepare_arguments(function.get_static_sig(), arguments);
  std::span<void*> pointers = prepared.buffer.get_ptrs();
  print_case_profile(
      benchmark_name, function.get_static_sig(), prepared.full_signature, pointers.size());
  Backend::ensure_context();
  const int device_index = Backend::get_device_index();
  const auto& kernel =
      function.get_or_compile_kernel(prepared.full_signature, options, device_index);

  auto device_launch = [&]() {
    std::apply(
        [&](const auto&... args) {
          function.launch_on_device(device_index, stream, 1, 1, 1, options, args...);
        },
        arguments);
  };

  const auto kernel_handle = Backend::load_kernel(kernel.get_dir(), kernel.get_kernel_name());
  const unsigned int shared_memory =
      Backend::get_shared_memory(kernel.get_dir(), kernel.get_kernel_name());
  const auto launch_options = Backend::prepare_launch(kernel.get_dir(),
                                                      kernel.get_kernel_name(),
                                                      shared_memory,
                                                      prepared.full_signature,
                                                      pointers.size());
  const unsigned int warp_size = get_warp_size(kernel);

  auto kernel_launch = [&]() {
    kernel.launch_with_signature(1,
                                 1,
                                 1,
                                 kNumWarps,
                                 stream,
                                 pointers.data(),
                                 prepared.full_signature,
                                 pointers.size());
  };
  auto raw_launch = [&]() {
    Backend::launch_kernel(stream,
                           kernel_handle,
                           1,
                           1,
                           1,
                           kNumWarps * warp_size,
                           1,
                           1,
                           pointers.data(),
                           launch_options);
  };
  auto argument_processing = [&]() {
    PreparedArguments current = prepare_arguments(function.get_static_sig(), arguments, false);
    std::span<void*> current_pointers = current.buffer.get_ptrs();
    benchmark_sink += current.buffer.size() + current.signature.size() + current.signature.hash() +
                      current_pointers.size() + reinterpret_cast<std::uintptr_t>(current_pointers.data());
  };

  validate_launch(benchmark_name,
                  "kernel_launch",
                  output_tensor,
                  expected,
                  kernel_launch);
  validate_launch(benchmark_name,
                  "operator_on_device",
                  output_tensor,
                  expected,
                  device_launch);
  validate_launch(benchmark_name,
                  "raw_backend",
                  output_tensor,
                  expected,
                  raw_launch);

  print_result(benchmark_name, "operator", measure_us(full_launch));
  print_result(benchmark_name, "operator_on_device", measure_us(device_launch));
  print_result(benchmark_name, "kernel_launch", measure_us(kernel_launch));
  print_result(benchmark_name, "raw_backend", measure_us(raw_launch));
  print_result(benchmark_name, "arguments_only", measure_us(argument_processing));
}

template <size_t NumArgs>
void run_pointer_kernel_benchmark(const std::filesystem::path& fixture,
                                  const std::string& benchmark_name,
                                  const std::string& function_name,
                                  at::Tensor& output_tensor,
                                  triton_jit::TritonDevicePtr output,
                                  triton_jit::TritonDevicePtr input) {
  run_kernel_benchmark(fixture,
                       benchmark_name,
                       function_name,
                       output_tensor,
                       make_argument_tuple<NumArgs>(output, input),
                       static_cast<float>(NumArgs - 1));
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cerr << "No CUDA-dispatch device is available\n";
    return 1;
  }

  const c10::Device benchmark_device {c10::DeviceType::CUDA, 0};
  const c10::DeviceGuard device_guard {benchmark_device};
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(benchmark_device);
  at::Tensor output_tensor = torch::zeros({1}, options);
  at::Tensor input_tensor = torch::ones({2}, options);
  const auto output = triton_jit::device_ptr(output_tensor.data_ptr<float>());
  const auto aligned_input = triton_jit::device_ptr(input_tensor.data_ptr<float>());
  const auto misaligned_input = triton_jit::device_ptr(input_tensor.data_ptr<float>() + 1);

  const std::filesystem::path fixture =
      std::filesystem::path {TRITON_JIT_TEST_SOURCE_DIR} / "fixtures" / "bench_kernels.py";

  std::cout << "Host launch overhead benchmark\n"
            << "warmup=" << kWarmupIterations << ", iterations=" << kBatchIterations
            << ", samples=" << kSamples << " (median)\n\n"
            << std::left << std::setw(kKernelColumnWidth) << "kernel"
            << std::setw(kLevelColumnWidth) << "level" << std::right << std::setw(10)
            << "us/call" << '\n';

  run_pointer_kernel_benchmark<4>(
      fixture, "launch_4", "launch_4", output_tensor, output, aligned_input);
  run_pointer_kernel_benchmark<11>(
      fixture, "launch_11", "launch_11", output_tensor, output, aligned_input);
  run_pointer_kernel_benchmark<30>(
      fixture, "launch_30", "launch_30", output_tensor, output, aligned_input);
  run_pointer_kernel_benchmark<30>(fixture,
                                   "specialized_misaligned_30",
                                   "launch_30",
                                   output_tensor,
                                   output,
                                   misaligned_input);
  run_pointer_kernel_benchmark<30>(fixture,
                                   "dynamic_nospec_30",
                                   "launch_dynamic_nospec_30",
                                   output_tensor,
                                   output,
                                   aligned_input);
  run_kernel_benchmark(fixture,
                       "constexpr_heavy_30",
                       "launch_constexpr_heavy_30",
                       output_tensor,
                       make_constexpr_heavy_argument_tuple(output),
                       435.0F);
  run_kernel_benchmark(fixture,
                       "mixed_30",
                       "launch_mixed_30",
                       output_tensor,
                       make_mixed_argument_tuple(output, aligned_input),
                       134.0F);
  return benchmark_sink == 0 ? 1 : 0;
}
