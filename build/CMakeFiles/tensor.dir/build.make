# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/valentin/coding/DeepRL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/coding/DeepRL/build

# Include any dependencies generated for this target.
include CMakeFiles/tensor.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tensor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tensor.dir/flags.make

CMakeFiles/tensor.dir/tensor.cpp.o: CMakeFiles/tensor.dir/flags.make
CMakeFiles/tensor.dir/tensor.cpp.o: ../tensor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/valentin/coding/DeepRL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tensor.dir/tensor.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tensor.dir/tensor.cpp.o -c /home/valentin/coding/DeepRL/tensor.cpp

CMakeFiles/tensor.dir/tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tensor.dir/tensor.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/valentin/coding/DeepRL/tensor.cpp > CMakeFiles/tensor.dir/tensor.cpp.i

CMakeFiles/tensor.dir/tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tensor.dir/tensor.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/valentin/coding/DeepRL/tensor.cpp -o CMakeFiles/tensor.dir/tensor.cpp.s

# Object files for target tensor
tensor_OBJECTS = \
"CMakeFiles/tensor.dir/tensor.cpp.o"

# External object files for target tensor
tensor_EXTERNAL_OBJECTS =

tensor: CMakeFiles/tensor.dir/tensor.cpp.o
tensor: CMakeFiles/tensor.dir/build.make
tensor: ../libtorch/lib/libtorch.so
tensor: ../libtorch/lib/libc10.so
tensor: ../libtorch/lib/libkineto.a
tensor: ../libtorch/lib/libc10.so
tensor: CMakeFiles/tensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/valentin/coding/DeepRL/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tensor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tensor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tensor.dir/build: tensor

.PHONY : CMakeFiles/tensor.dir/build

CMakeFiles/tensor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tensor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tensor.dir/clean

CMakeFiles/tensor.dir/depend:
	cd /home/valentin/coding/DeepRL/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/coding/DeepRL /home/valentin/coding/DeepRL /home/valentin/coding/DeepRL/build /home/valentin/coding/DeepRL/build /home/valentin/coding/DeepRL/build/CMakeFiles/tensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tensor.dir/depend

