CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(PA1-2 LANGUAGES CXX)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

ADD_SUBDIRECTORY(deps/vecmath)

SET(PA1_SOURCES
        src/image.cpp
        src/main.cpp
        src/mesh.cpp
        src/scene_parser.cpp
        src/texture.cpp)

SET(PA1_INCLUDES
        include/camera.hpp
        include/group.hpp
        include/hit.hpp
        include/image.hpp
        include/light.hpp
        include/material.hpp
        include/texture.hpp
        include/raytracer.hpp
        include/mesh.hpp
        include/object3d.hpp
        include/plane.hpp
        include/ray.hpp
        include/scene_parser.hpp
        include/sphere.hpp
        include/transform.hpp
        include/triangle.hpp
        )

SET(CMAKE_CXX_STANDARD 11)

INCLUDE(CheckLanguage)
CHECK_LANGUAGE(CUDA)
IF(CMAKE_CUDA_COMPILER)
    ENABLE_LANGUAGE(CUDA)
    LIST(APPEND PA1_SOURCES src/cuda_scene_builder.cpp src/bvh_builder.cpp src/cuda_path_tracer.cu)
    LIST(APPEND PA1_INCLUDES include/cuda_renderer.hpp include/cuda_types.h)
    MESSAGE(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
ELSE()
    MESSAGE(STATUS "CUDA compiler not found; building CPU-only")
ENDIF()

ADD_EXECUTABLE(${PROJECT_NAME} ${PA1_SOURCES} ${PA1_INCLUDES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE include)

IF(CMAKE_CUDA_COMPILER)
    TARGET_COMPILE_DEFINITIONS(${PROJECT_NAME} PRIVATE USE_CUDA=1)
    SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        SET(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    ENDIF()
    FIND_PACKAGE(CUDAToolkit REQUIRED)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} CUDA::cudart CUDA::curand)
    SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "89")
ENDIF()

ADD_EXECUTABLE(gen_textures src/gen_textures.cpp src/texture.cpp)
TARGET_LINK_LIBRARIES(gen_textures vecmath)
TARGET_INCLUDE_DIRECTORIES(gen_textures PRIVATE include)
TARGET_COMPILE_OPTIONS(gen_textures PRIVATE -O2)

FIND_PACKAGE(OpenMP)
IF(OpenMP_CXX_FOUND)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    MESSAGE(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
ELSE()
    MESSAGE(STATUS "OpenMP not found; build is serial-only (pass omp/parallel flag has no effect)")
ENDIF()
