# The AMReX drivers of the patchwork vs AMReX benchmarks, built against an installed AMReX
# (one build per dimension): cmake -S . -B build2 -DAMReX_DIR=<amrex-install2>/lib/cmake/AMReX
cmake_minimum_required(VERSION 3.20)
project(patchwork_amrex_benchmarks C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(AMReX REQUIRED)
foreach(driver bench_solve bench_mesh bench_particles)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${driver}.cpp)
    add_executable(${driver} ${driver}.cpp)
    target_link_libraries(${driver} PRIVATE AMReX::amrex)
  endif()
endforeach()
