cmake_minimum_required(VERSION 2.8)
project( connect_components_analysis )
find_package( OpenCV REQUIRED )

include_directories (
  "${PROJECT_SOURCE_DIR}"/include/connected.h
  "${PROJECT_SOURCE_DIR}"/include/contour_thinning.h)

add_executable(
  main src/main.cpp
  #connected.cpp
  )

target_link_libraries( main ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR} )