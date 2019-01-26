cmake_minimum_required(VERSION 3.9)
project(YandexOptimization)

set(CMAKE_CXX_STANDARD 14)


# SET(CMAKE_CXX_FLAGS "-Wno-sign-compare -DLOCAL -Wall ")
if (CMAKE_BUILD_TYPE MATCHES TOSUBMIT)
    SET(CMAKE_CXX_FLAGS "-O2 -Wno-sign-compare -DTOSUBMIT -Wall ")
endif()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    SET(CMAKE_CXX_FLAGS "-Wno-sign-compare -DTEST -DLOCAL -Wall ")
endif()
if (CMAKE_BUILD_TYPE MATCHES Release)
    SET(CMAKE_CXX_FLAGS "-O2 -Wno-sign-compare -DTEST -DLOCAL -Wall ")
endif()

add_executable(YandexOptimization main.cpp )
