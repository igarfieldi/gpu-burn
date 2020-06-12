cmake_minimum_required(VERSION 3.12)
project("gpuburn" VERSION 0.0.1 LANGUAGES CXX CUDA)

# Discourage in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(WARNING "In-source building is discouraged!")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

# We need CUDA
find_package(CUDA 9.1 REQUIRED)

# Find the NVML library (taken from https://tianyuliukingcrimson.com/2018/07/23/findnvml-cmake-done-correctly-how-to-have-cmake-find-nvidia-management-library-nvml-on-windows-and-linux/)
if(WIN32)
    # Windows, including both 32-bit and 64-bit
    set(NVML_NAMES nvml)
    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
 
    # .lib import library full path
    find_file(NVML_LIB_PATH
              NO_DEFAULT_PATH
              NAMES nvml.lib
              PATHS ${NVML_LIB_DIR})
 
    # .dll full path
    find_file(NVML_DLL_PATH
              NO_DEFAULT_PATH
              NAMES nvml.dll
              PATHS "C:/Program Files/NVIDIA Corporation/NVSMI")
elseif(UNIX AND NOT APPLE)
    # Linux
    set(NVML_NAMES nvidia-ml)
    set(NVML_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs")
    set(NVML_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
 
    find_library(NVML_LIB_PATH
                 NO_DEFAULT_PATH
                 NAMES ${NVML_NAMES}
                 PATHS ${NVML_LIB_DIR})
else()
    message(FATAL_ERROR "Unsupported platform.")
endif()
 
find_path(NVML_INCLUDE_PATH
          NO_DEFAULT_PATH
          NAMES nvml.h
          PATHS ${NVML_INCLUDE_DIR})
 
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG NVML_LIB_PATH NVML_INCLUDE_PATH)

# CUDA arch
set(GPUBURN_CUDA_ARCH "Auto" CACHE STRING "CUDA target compute capability (Auto uses present GPUs")

# Select the compilation target for NVCC in terms of compute capability
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${GPUBURN_CUDA_ARCH})
if("${GPUBURN_CUDA_ARCH}" STREQUAL "Auto")
	STRING(REGEX REPLACE "sm_(.)(.)" "\\1.\\2" ARCH_CC ${ARCH_FLAGS_readable})
	set(GPUBURN_CUDA_ARCH "${ARCH_CC}" CACHE STRING "CUDA target compute capability (Auto uses present GPUs)" FORCE)
	message(STATUS "Detected CUDA arch: ${GPUBURN_CUDA_ARCH}")
endif()
# For whatever reason the arch flags may be in the wrong format, so we gotta fix that
string(REGEX REPLACE "gencode;" "gencode=" ARCH_FLAGS "${ARCH_FLAGS}")
string(REGEX REPLACE ";code" ",code" ARCH_FLAGS "${ARCH_FLAGS}")
string(REGEX REPLACE ";-gencode" " -gencode" ARCH_FLAGS "${ARCH_FLAGS}")
# Since CMake's target_link_options does not pass options to nvlink yet, we have to set them globally to the NVCC flags...
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")

# Ensure that our binaries get put in the proper directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build/$<CONFIG>")


add_executable(gpuburn gpu_burn-drv.cu)
set_target_properties(gpuburn
	PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
		CXX_EXTENSIONS OFF
		CUDA_SEPARABLE_COMPILATION OFF
		CUDA_STANDARD 17
		CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(gpuburn PRIVATE ${CUDA_CUBLAS_LIBRARIES} ${NVML_LIB_PATH})

# Make sure to copy DLL into executable path
if(NVML_DLL_PATH)
	add_custom_command(TARGET gpuburn PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${NVML_DLL_PATH}"
			"${CMAKE_BINARY_DIR}/build/$<CONFIG>"
	)
endif(NVML_DLL_PATH)