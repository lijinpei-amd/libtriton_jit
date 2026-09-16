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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "triton_jit/kernel_metadata.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("libtriton_jit_amdgpu_metadata_" + std::to_string(nonce));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void write_metadata(const std::filesystem::path& directory,
                    const std::string& kernel_name,
                    const std::string& contents) {
  std::ofstream output(directory / (kernel_name + ".json"));
  require(output.good(), "failed to create metadata fixture");
  output << contents;
}

}  // namespace

int main() {
  TemporaryDirectory temporary;

  write_metadata(temporary.path(),
                 "wave32_kernel",
                 R"({
                   "shared": 4096,
                   "name": "wave32_symbol",
                   "warp_size": 32,
                   "num_ctas": 1,
                   "launch_cooperative_grid": true,
                   "global_scratch_size": 0,
                   "profile_scratch_size": 0,
                   "triton_version": "3.5.0",
                   "target": {
                     "backend": "hip",
                     "arch": "gfx1100",
                     "warp_size": 32
                   }
                 })");
  triton_jit::AmdgpuKernelMetadata wave32 =
      triton_jit::load_amdgpu_metadata(temporary.path().string(), "wave32_kernel");
  require(wave32.shared == 4096, "shared memory was not parsed");
  require(wave32.symbol_name == "wave32_symbol", "kernel symbol was not parsed");
  require(wave32.warp_size == 32, "wave32 was not parsed");
  require(wave32.num_ctas == 1, "num_ctas was not parsed");
  require(wave32.arch == "gfx1100", "RDNA architecture was not parsed");
  require(wave32.target_backend == "hip", "HIP target backend was not parsed");
  require(wave32.launch_cooperative_grid, "cooperative launch flag was not parsed");
  require(wave32.triton_version == "3.5.0", "Triton version was not parsed");

  write_metadata(temporary.path(),
                 "wave64_kernel",
                 R"({
                   "shared": 65536,
                   "target": {
                     "backend": "hip",
                     "arch": "gfx942",
                     "warp_size": 64
                   }
                 })");
  triton_jit::AmdgpuKernelMetadata wave64 =
      triton_jit::load_amdgpu_metadata(temporary.path().string(), "wave64_kernel");
  require(wave64.warp_size == 64, "target warp_size fallback was not parsed");
  require(wave64.arch == "gfx942", "CDNA architecture was not parsed");

  std::cout << "AMDGPU metadata tests passed\n";
  return 0;
}
