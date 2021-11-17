cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
set(OpenGL_GL_PREFERENCE "GLVND")
#set(CMAKE_CXX_FLAGS "-fsanitize=thread")
find_package(OpenGL REQUIRED)


include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC})

file(GLOB CSC4005_PROJECT_SRC src/main_seq.cpp src/graphic.cpp)
file(GLOB CSC4005_PROJECT_SRC_MPI src/main_MPI.cpp src/graphic.cpp)
file(GLOB CSC4005_PROJECT_SRC_PTHREAD src/main_Pthread.cpp src/graphic.cpp)
#file(GLOB CSC4005_PROJECT_SRC_CUDA src/main.cu src/graphic.cpp)
file(GLOB CSC4005_PROJECT_SRC_OMP src/main_openmp.cpp src/graphic.cpp)
file(GLOB CSC4005_PROJECT_SRC_BOS src/main_BS.cpp src/graphic.cpp)


add_executable(csc4005_imgui ${CSC4005_PROJECT_SRC})
add_executable(testMPI ${CSC4005_PROJECT_SRC_MPI})
add_executable(testP ${CSC4005_PROJECT_SRC_PTHREAD})
#add_executable(testCUDA ${CSC4005_PROJECT_SRC_CUDA})
add_executable(testOMP ${CSC4005_PROJECT_SRC_OMP})
add_executable(testBS ${CSC4005_PROJECT_SRC_BOS})



get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})

target_link_libraries(csc4005_imgui core)
target_compile_options(csc4005_imgui PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(testMPI core)
target_compile_options(testMPI PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(testP core)
target_compile_options(testP PRIVATE -Werror -Wall -Wextra -Wpedantic)
#target_link_libraries(testCUDA core)
#target_compile_options(testCUDA PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(testOMP core)
target_compile_options(testOMP PRIVATE
          $<$<COMPILE_LANGUAGE:C>:${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}>)
target_compile_options(testOMP PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_options(testOMP PRIVATE ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp=libomp)

target_link_libraries(testBS core)
target_compile_options(testBS PRIVATE
          $<$<COMPILE_LANGUAGE:C>:${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}>
          $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}>)
target_compile_options(testBS PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_options(testBS PRIVATE ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp=libomp)

target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)

target_compile_definitions(csc4005_imgui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(testMPI PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(testP PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
#target_compile_definitions(testCUDA PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(testOMP PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(testBS PRIVATE -DFONT_PATH=\"${FONT_PATH}\")


if (WIN32)
    target_link_options(csc4005_imgui BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(testMPI BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(testP BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    #target_link_options(testCUDA BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(testOMP BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(testBS BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()