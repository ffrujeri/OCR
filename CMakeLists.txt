CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(CMAKE_MODULE_PATH $ENV{OPENCV_DIR})

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(modal_OCR)

ADD_EXECUTABLE(ocr ocr.cpp)

TARGET_LINK_LIBRARIES(ocr ${OpenCV_LIBS})
