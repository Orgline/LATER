# LATER
Linear Algebra on TEnsoRcore

## Prerequisites

* CMake 3.12+
* CUDA 10.1+

## Build

```
$ mkdir build && cd build
$ export CUDACXX=/usr/local/cuda-10.1/bin/nvcc
$ export CUDA_PATH=/usr/local/cuda-10.1
$ cmake ..
```

Change the CUDACXX and CUDA_PATH environment variables to match
your system's CUDA installation directory. 

## Run tests

$ cd QR
$ ./test_qr 1 16384 16384

## Tested GPUs and Platforms
* V100 (on RHEL Linux, CUDA 10.1)
* Titan V (on Ubuntu 18.04 Linux, CUDA 10.1)
* GeForce RTX 2060 (on Windows 10, CUDA 10.2)