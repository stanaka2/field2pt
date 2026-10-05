cmake_minimum_required(VERSION 3.18)
project(field2pt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Development.Module rather than the full library: an extension module does not link libpython.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

find_path(FFTW_INCLUDE_DIR fftw3.h)
find_library(FFTW_DOUBLE_LIB fftw3)
find_library(FFTW_FLOAT_LIB fftw3f)
find_library(FFTW_DOUBLE_OMP_LIB fftw3_omp)
find_library(FFTW_FLOAT_OMP_LIB fftw3f_omp)

set(FFTW_HINT
  "Install it with 'apt install libfftw3-dev' (Debian, Ubuntu) or 'dnf install fftw-devel' (RHEL, "
  "Fedora), or point CMAKE_PREFIX_PATH at an installation elsewhere, such as the one a "
  "'module load fftw' provides.")
if(NOT FFTW_INCLUDE_DIR OR NOT FFTW_DOUBLE_LIB OR NOT FFTW_FLOAT_LIB)
  message(FATAL_ERROR "FFTW3 (single and double precision) is required. " ${FFTW_HINT})
endif()
if(NOT FFTW_DOUBLE_OMP_LIB OR NOT FFTW_FLOAT_OMP_LIB)
  message(FATAL_ERROR "the OpenMP threaded FFTW3 libraries are required. " ${FFTW_HINT})
endif()

# An FFTW found outside the system directories, through CMAKE_PREFIX_PATH or a module, is recorded
# in the installed module, so that it still loads once LD_LIBRARY_PATH no longer names it.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

pybind11_add_module(_binding MODULE src/binding.cpp)

target_include_directories(_binding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${FFTW_INCLUDE_DIR})
target_link_libraries(_binding PRIVATE
  OpenMP::OpenMP_CXX
  ${FFTW_FLOAT_OMP_LIB} ${FFTW_FLOAT_LIB}
  ${FFTW_DOUBLE_OMP_LIB} ${FFTW_DOUBLE_LIB}
)

# -ffp-contract=off keeps |k| from being computed with a fused multiply-add. A mode that sits
# exactly on a bin edge would otherwise land on either side depending on the contraction the
# compiler chose, which makes the mode counts irreproducible.
target_compile_options(_binding PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native;-ffp-contract=off>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
)


# ---- install ----
install(TARGETS _binding LIBRARY DESTINATION field2pt)
install(DIRECTORY python/ DESTINATION field2pt FILES_MATCHING PATTERN "*.py")
