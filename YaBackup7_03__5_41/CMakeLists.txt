cmake_minimum_required(VERSION 3.9)
project(YandexOptimization)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "-O2 -Wno-sign-compare -DTEST -DLOCAL -Wall ")
add_executable(YandexOptimization main.cpp)