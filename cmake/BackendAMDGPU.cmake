# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================
# AMDGPU / ROCm Backend Configuration
# ==============================================================================

message(STATUS "Configuring AMDGPU backend...")

# This module runs after find_package(Torch). PyTorch's LoadHIP.cmake has already
# resolved the ROCm tree (honouring ENV{ROCM_PATH}, else /opt/rocm) and has already
# run find_package(hip CONFIG), leaving ROCM_PATH, hip_VERSION and hip_INCLUDE_DIRS
# in this scope. Align with that result rather than searching independently: the HIP
# headers we compile against must describe the runtime PyTorch will load.
if(NOT PYTORCH_FOUND_HIP)
    message(FATAL_ERROR
        "BACKEND=AMDGPU requires a ROCm-enabled PyTorch installation, but PyTorch's "
        "CMake package did not configure HIP. Point the ROCM_PATH environment variable "
        "at the ROCm installation root before configuring.")
endif()

if(NOT ROCM_PATH OR NOT IS_DIRECTORY "${ROCM_PATH}")
    message(FATAL_ERROR
        "PyTorch resolved the ROCm root to '${ROCM_PATH}', which is not a directory.")
endif()
# There is deliberately no knob to select a ROCm tree here. The backend must use the
# one PyTorch resolved, so ENV{ROCM_PATH} is the single point of control for both.
file(REAL_PATH "${ROCM_PATH}" AMDGPU_ROCM_ROOT)
message(STATUS "ROCm root (from PyTorch): ${AMDGPU_ROCM_ROOT}")

if(NOT hip_FOUND)
    message(FATAL_ERROR
        "PyTorch configured HIP, but the hip CMake package was not found in this scope")
endif()
if(NOT hip_INCLUDE_DIRS)
    message(FATAL_ERROR "The ROCm HIP package did not provide its include directory")
endif()
message(STATUS "Found ROCm HIP: ${hip_VERSION}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch; value = torch.version.hip; print(value if value is not None else '')"
    RESULT_VARIABLE AMDGPU_TORCH_CHECK_RESULT
    OUTPUT_VARIABLE AMDGPU_TORCH_HIP_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE AMDGPU_TORCH_CHECK_ERROR
)
if(NOT AMDGPU_TORCH_CHECK_RESULT EQUAL 0 OR AMDGPU_TORCH_HIP_VERSION STREQUAL "")
    message(FATAL_ERROR
        "BACKEND=AMDGPU requires a ROCm-enabled PyTorch installation. "
        "The selected Python is ${Python_EXECUTABLE}. ${AMDGPU_TORCH_CHECK_ERROR}")
endif()
message(STATUS "PyTorch ROCm version: ${AMDGPU_TORCH_HIP_VERSION}")

string(REGEX MATCH "^[0-9]+\\.[0-9]+" AMDGPU_ROCM_MAJOR_MINOR "${hip_VERSION}")
string(REGEX MATCH "^[0-9]+\\.[0-9]+" AMDGPU_TORCH_MAJOR_MINOR "${AMDGPU_TORCH_HIP_VERSION}")
if(NOT AMDGPU_ROCM_MAJOR_MINOR OR NOT AMDGPU_TORCH_MAJOR_MINOR)
    message(FATAL_ERROR
        "Unable to compare ROCm HIP version '${hip_VERSION}' with PyTorch HIP version "
        "'${AMDGPU_TORCH_HIP_VERSION}'")
endif()
if(NOT AMDGPU_ROCM_MAJOR_MINOR VERSION_EQUAL AMDGPU_TORCH_MAJOR_MINOR)
    message(FATAL_ERROR
        "${AMDGPU_ROCM_ROOT} provides ROCm HIP ${hip_VERSION}, but PyTorch was built for "
        "HIP ${AMDGPU_TORCH_HIP_VERSION}. Point ROCM_PATH at a matching ROCm "
        "${AMDGPU_TORCH_MAJOR_MINOR} installation to avoid loading incompatible HIP "
        "components.")
endif()

# This target intentionally provides headers and the dynamic-loader library,
# but does not link hip::host. The launcher resolves HIP calls from the runtime
# already loaded by PyTorch, which prevents bundled and system HIP runtimes from
# coexisting in one process.
if(NOT TARGET TritonJIT::amdgpu_host_api)
    add_library(TritonJIT::amdgpu_host_api INTERFACE IMPORTED GLOBAL)
    set_target_properties(TritonJIT::amdgpu_host_api PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1"
        INTERFACE_COMPILE_FEATURES "cxx_std_20"
        INTERFACE_INCLUDE_DIRECTORIES "${hip_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS}")
endif()

function(target_link_amdgpu_libraries target_name)
    target_link_libraries(${target_name} PRIVATE TritonJIT::amdgpu_host_api)
    target_compile_features(${target_name} PUBLIC cxx_std_20)
endfunction()

message(STATUS "AMDGPU backend configuration complete")
