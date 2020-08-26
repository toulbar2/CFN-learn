#!/bin/sh

git submodule init
git submodule update
cd tb2-submodule
mkdir build
cd build
cmake -DPYTB2=ON -DXML=ON ..
make -j $(python3 -c 'import multiprocessing as m; print(m.cpu_count())')
cd ../..
ln -s tb2-submodule/build/lib/*/pytoulbar2.*

