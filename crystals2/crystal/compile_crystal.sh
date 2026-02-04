#!/bin/bash

# Compile all crystal simulation C++ files
# Requires: libfltk1.3-dev, libgl1-mesa-dev, libglu1-mesa-dev

echo "Compiling crystal_growth.cpp..."
g++ crystal_growth.cpp -o crystal_growth -lfltk -lfltk_gl -lGL -lGLU -O3

echo "Compiling electro.cpp..."
g++ electro.cpp -o electro -lfltk -lfltk_gl -lGL -lGLU -O3

echo "Compiling electro_plating_enhanced.cpp..."
g++ electro_plating_enhanced.cpp -o electro_plating_enhanced -lfltk -lfltk_gl -lGL -lGLU -O3

echo "Compiling realistic.cpp..."
g++ realistic.cpp -o realistic -lfltk -lfltk_gl -lGL -lGLU -O3

echo "Done."
