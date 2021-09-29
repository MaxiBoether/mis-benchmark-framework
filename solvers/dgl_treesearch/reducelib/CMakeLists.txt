cmake_minimum_required(VERSION 3.19)
project(DGLReducelib)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(./KaMIS)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP support detected")
  add_definitions(${OpenMP_CXX_FLAGS})
else()
  message(WARNING "OpenMP not available, activating workaround")
  add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/misc)
endif()

add_library(reduceu SHARED reducelib.cpp)
target_include_directories(reduceu PRIVATE
    KaMIS/lib/mis/kernel
    KaMIS/lib/mis
    KaMIS/extern/KaHIP/lib
    KaMIS/extern/KaHIP/lib/tools
    KaMIS/lib/mis/evolutionary
    KaMIS/lib
    KaMIS/extern/KaHIP/lib/data_structure
    KaMIS/lib/mis/initial_mis
    KaMIS/extern/KaHIP/lib/partition
    KaMIS/app
    KaMIS/lib/mis/ils
    KaMIS/extern/KaHIP/interface
    )
target_compile_options(reduceu PUBLIC -O3 -Wall)
target_link_libraries(reduceu PRIVATE libfiles libkaffpa ${OpenMP_CXX_LIBRARIES})

add_library(reducew SHARED reducelib_weighted.cpp)
target_include_directories(reducew PRIVATE
    ./wmispatch
    KaMIS/wmis/lib/mis/kernel
    KaMIS/wmis/lib/mis
    KaMIS/wmis/extern/KaHIP/lib
    KaMIS/wmis/extern/KaHIP/lib/tools
    KaMIS/wmis/lib
    KaMIS/wmis/extern/KaHIP/lib/data_structure
    KaMIS/wmis/lib/mis/ils
    )
target_compile_options(reducew PUBLIC -O3 -Wall)
target_link_libraries(reducew PRIVATE libsources libkaffpa2 ${OpenMP_CXX_LIBRARIES})

# patch KaMIS branch_and_reduce_algorithm
target_include_directories(libsources BEFORE PRIVATE ./wmispatch)