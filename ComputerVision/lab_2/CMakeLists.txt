cmake_minimum_required(VERSION 2.8)
project(lab_2)

# Найти пакет OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Включить директорию с заголовочными файлами
include_directories(${CMAKE_SOURCE_DIR})

# Создать исполняемый файл
add_executable(lab_2 lab_2.cpp imageGenerator.cpp)

# Привязать библиотеки OpenCV
target_link_libraries(coursework
    ${OpenCV_LIBS}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_features2d
)
