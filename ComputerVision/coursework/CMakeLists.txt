cmake_minimum_required(VERSION 2.8)
project(coursework)

# Найти пакет OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Создать исполняемый файл
add_executable(coursework coursework.cpp)

target_link_libraries(coursework
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_features2d
)
