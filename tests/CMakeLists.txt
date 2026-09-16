if(BUILD_TESTING)
add_executable(test_launch_hooks test_launch_hooks.cpp)
target_link_libraries(test_launch_hooks PRIVATE TritonJIT::triton_jit)
add_test(NAME launch_hooks COMMAND test_launch_hooks)

add_executable(test_shared_memory_cache test_shared_memory_cache.cpp)
target_link_libraries(test_shared_memory_cache PRIVATE TritonJIT::triton_jit)
add_test(NAME shared_memory_cache COMMAND test_shared_memory_cache)

add_executable(test_runtime_concurrency test_runtime_concurrency.cpp)
target_link_libraries(test_runtime_concurrency PRIVATE TritonJIT::triton_jit)
add_test(NAME runtime_concurrency COMMAND test_runtime_concurrency)

add_executable(test_parameter_buffer test_parameter_buffer.cpp)
target_link_libraries(test_parameter_buffer PRIVATE TritonJIT::triton_jit)
add_test(NAME parameter_buffer COMMAND test_parameter_buffer)

add_executable(test_explicit_device_launch test_explicit_device_launch.cpp)
target_link_libraries(test_explicit_device_launch PRIVATE TritonJIT::triton_jit)
add_test(NAME explicit_device_launch COMMAND test_explicit_device_launch)

add_executable(test_tuple_args test_tuple_args.cpp)
target_link_libraries(test_tuple_args PRIVATE TritonJIT::triton_jit)

add_test(NAME tuple_args_cpp COMMAND test_tuple_args)
add_test(
    NAME tuple_signature_python
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_tuple_signature.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
)

add_executable(test_jit_function_arg test_jit_function_arg.cpp)
target_link_libraries(test_jit_function_arg PRIVATE TritonJIT::triton_jit)
target_compile_definitions(
    test_jit_function_arg
    PRIVATE TRITON_JIT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME jit_function_arg_cpp COMMAND test_jit_function_arg)
add_test(
    NAME jit_function_arg_python
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_jit_function.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/jit_function_module.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/jit_function_compile_module.py
)

add_executable(test_device_ptr test_device_ptr.cpp)
target_link_libraries(test_device_ptr PRIVATE TritonJIT::triton_jit)
add_test(NAME device_ptr_cpp COMMAND test_device_ptr)

add_test(
    NAME standalone_equal_to_one
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_equal_to_one.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/equal_to_one_module.py
)

if(BACKEND STREQUAL "AMDGPU")
    add_executable(test_amdgpu_metadata test_amdgpu_metadata.cpp)
    target_link_libraries(test_amdgpu_metadata PRIVATE TritonJIT::triton_jit)
    add_test(NAME amdgpu_metadata COMMAND test_amdgpu_metadata)
endif()
endif()

# The launch-overhead benchmark currently requires a CUDA-dispatch backend.
if(BUILD_TESTING AND BACKEND MATCHES "^(AMDGPU|CUDA|HCU|IX)$")
    add_executable(bench_launch bench_launch.cpp)
    target_link_libraries(bench_launch PRIVATE TritonJIT::triton_jit)
    target_compile_definitions(
        bench_launch
        PRIVATE TRITON_JIT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()
