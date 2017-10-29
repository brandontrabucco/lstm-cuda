#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/../src
nvcc  -std=c++11 -L/usr/local/cuda/lib64 -lcuda -lcudart -dc *.cu
if [ ! -d $baseDir/../build ]; then
    rm -rf $baseDir/../build
    mkdir $baseDir/../build
fi
tar cf - .|(cd $baseDir/../build; tar xf -) 
cd $baseDir/../build
nvcc *.o -o lstm
echo "path:" $baseDir/../build/lstm
# <learning rate> <decay rate> <blocks> <cells> <size ...>
./lstm 0.01 0.01 10 100 10
