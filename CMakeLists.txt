cmake_minimum_required(VERSION 3.10)
project(depthInpainting)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Source files
file(GLOB_RECURSE SRC 
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")

add_executable(depthInpainting ${SRC})

# Include and link
target_include_directories(depthInpainting PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(depthInpainting PRIVATE
    ${OpenCV_LIBS})

# Build settings
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
