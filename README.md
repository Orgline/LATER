# LATER
Linear Algebra on TEnsoRcore

## Prerequisites

* CMake 3.12+
* CUDA 10.1+

## Build
On Linux, 
```
$ git clone git@github.com:Orgline/LATER.git
$ cd LATER && mkdir build && cd build
$ export CUDACXX=/usr/local/cuda-10.1/bin/nvcc
$ export CUDA_PATH=/usr/local/cuda-10.1
$ cmake ..
$ cmake --build .
```
Change the CUDACXX and CUDA_PATH environment variables to match
your system's CUDA installation directory. 


On Windows, 

```
$ git clone git@github.com:Orgline/LATER.git
$ cd LATER && mkdir build && cd build
$ cmake .. -A x64
$ cmake --build .
```
## Run tests
On Linux
```
$ cd test
$ ./test_qr 1 16384 16384 -check
```

On Windows:
```
$ cd test/debug
$ test_qr.exe 1 16384 16384 -check
```

## Tested GPUs and Platforms
* V100 (on RHEL Linux 7, CUDA 10.1, GCC 8)
* Titan V (on Ubuntu 18.04 Linux, CUDA 10.1, GCC 7.5.0)
* GeForce RTX 2060 (on Windows 10, CUDA 10.2, Visual Studio 2017)