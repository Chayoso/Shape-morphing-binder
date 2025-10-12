cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(DiffMPMLib LANGUAGES CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

add_library(DiffMPMLib STATIC
    pch.h
    BackPropagation.h
    BackPropagation.cpp
    cereal_eigen.h
    CompGraph.h
    CompGraph.cpp
    Elasticity.h
    Elasticity.cpp
    ForwardSimulation.h
    ForwardSimulation.cpp
    framework.h
    GeometryLoading.h
    GeometryLoading.cpp
    Grid.h
    Grid.cpp
    GridNode.h
    GridNode.cpp
    Interpolation.h
    MaterialPoint.h
    MaterialPoint.cpp
    PointCloud.h
    PointCloud.cpp
    SphereUnionSurfacing.h
    SphereUnionSurfacing.cpp
    Tensor3x3x3x3.h
    Tensor3x3x3x3.cpp
)

set_target_properties(DiffMPMLib PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(DiffMPMLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Release-like defaults and Eigen debug off
target_compile_definitions(DiffMPMLib PUBLIC EIGEN_NO_DEBUG NDEBUG)

if(MSVC)
  target_compile_options(DiffMPMLib PRIVATE /O2 /MP /permissive- /DNOMINMAX /EHsc)
  # Link-time optimization (LTO)
  target_compile_options(DiffMPMLib PRIVATE /GL)
  target_link_options(DiffMPMLib PRIVATE /LTCG)
else()
  target_compile_options(DiffMPMLib PRIVATE -O3)
  # optional LTO if toolchain supports it
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
  if(ipo_supported)
    set_property(TARGET DiffMPMLib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)