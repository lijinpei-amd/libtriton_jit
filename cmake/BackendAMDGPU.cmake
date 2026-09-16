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

set(AMDGPU_ROOT "" CACHE PATH "ROCm installation root for the AMDGPU backend")
if(NOT AMDGPU_ROOT)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(AMDGPU_ROOT "$ENV{ROCM_PATH}")
    elseif(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        set(AMDGPU_ROOT "$ENV{HIP_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(AMDGPU_ROOT "/opt/rocm")
    endif()
endif()

if(NOT AMDGPU_ROOT OR NOT IS_DIRECTORY "${AMDGPU_ROOT}")
    message(FATAL_ERROR
        "A valid ROCm installation is required for BACKEND=AMDGPU. "
        "Set AMDGPU_ROOT to the ROCm installation root.")
endif()

file(REAL_PATH "${AMDGPU_ROOT}" AMDGPU_ROOT)
set(AMDGPU_ROOT "${AMDGPU_ROOT}" CACHE PATH "ROCm installation root for the AMDGPU backend" FORCE)
set(ENV{ROCM_PATH} "${AMDGPU_ROOT}")
list(PREPEND CMAKE_PREFIX_PATH "${AMDGPU_ROOT}" "${AMDGPU_ROOT}/hip")
message(STATUS "AMDGPU_ROOT: ${AMDGPU_ROOT}")

find_package(hip CONFIG REQUIRED
    PATHS "${AMDGPU_ROOT}" "${AMDGPU_ROOT}/lib/cmake/hip"
    NO_DEFAULT_PATH)
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
        "AMDGPU_ROOT provides ROCm HIP ${hip_VERSION}, but PyTorch was built for HIP "
        "${AMDGPU_TORCH_HIP_VERSION}. Select a matching ROCm ${AMDGPU_TORCH_MAJOR_MINOR} "
        "installation to avoid loading incompatible HIP components.")
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
