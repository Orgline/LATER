# LATER
Linear Algebra on TEnsoRcore;
see http://www2.cs.uh.edu/~panruowu/later.html

## Prerequisites

* CMake 3.12+
* CUDA 10.1+
* CUTLASS 2.1+

## Build
On Linux, 
```
$ git clone git@github.com:Orgline/LATER.git
$ cd LATER && mkdir build && cd build
$ export CUDACXX=/usr/local/cuda-10.1/bin/nvcc
$ export CUDA_PATH=/usr/local/cuda-10.1
$ export CUTLASS_DIR=<CUTLASS Diretory> 
$ # for exmaple ~/cutlass-2.1.0
$ cmake .. -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_75,code=sm_75" -DCUDA_ARCH="Turing"
$ # On Volta, 75->70, Turing->Volta
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
* GeForce RTX 2080 Super (Ubuntu 18.04 Linux, CUDA 10.2, GCC 7.5.0)