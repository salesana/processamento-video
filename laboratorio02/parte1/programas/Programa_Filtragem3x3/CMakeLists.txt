cmake_minimum_required(VERSION 2.8)
project( Filtragem_3x3 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Filtragem_3x3 Filtragem_3x3.cpp )
target_link_libraries( Filtragem_3x3 ${OpenCV_LIBS} )
