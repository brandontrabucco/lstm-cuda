#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/..
nvcc  -std=c++11 -L/usr/local/cuda/lib64 -lcuda -lcudart -dc *.cu
if [ ! -d $baseDir/../build ]; then
    mkdir $baseDir/../build
fi
mv *.o $baseDir/../build
cd $baseDir/../build
nvcc *.o -o lstm
./lstm
