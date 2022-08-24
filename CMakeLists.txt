cmake_minimum_required(VERSION 3.18)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

project(parking_cpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

pkg_check_modules(ZMQ REQUIRED libzmq)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})


find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


option(DEBUG_LOG "Debug Mode" OFF)
option(IMSHOW "Show Images" OFF)
option(IMSAVE "Save Images" OFF)
option(MORE_LOGS "Log tracking points" OFF)

if (DEBUG_LOG)
    add_definitions(-DDEBUG=1)
else ()
    add_definitions(-DDEBUG=0)
endif ()

if (IMSHOW)
    add_definitions(-DIMSHOW=1)
else ()
    add_definitions(-DIMSHOW=0)
endif ()

if (MORE_LOGS)
    add_definitions(-DMORE_LOGS=1)
else ()
    add_definitions(-DMORE_LOGS=0)
endif ()

if (IMSAVE)
    add_definitions(-DIMSAVE=1)
else ()
    add_definitions(-DIMSAVE=0)
endif ()

add_executable(parking_cpu src/RunnerDetector.cpp)
target_include_directories(parking_cpu PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(parking_cpu ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS} ${ZMQ_LIBRARIES} ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES}
        nlohmann_json::nlohmann_json ${CURL_LIBRARIES} /usr/local/lib/libcpr.so)