cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tensor)
set(Torch_DIR "/home/valentin/coding/DeepRL/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#add_executable(tensor deepNN_gpt.cpp)
add_executable(tensor tensor.cpp)
add_executable(samples samples.cpp)
add_executable(testing testing.cpp)
#target_link_libraries(tensor "${TORCH_LIBRARIES}")
target_link_libraries(tensor "${TORCH_LIBRARIES}")
target_link_libraries(testing "${TORCH_LIBRARIES}")
target_link_libraries(testing "${TORCH_LIBRARIES}")
set_property(TARGET tensor PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET tensor
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:tensor>)
endif (MSVC)