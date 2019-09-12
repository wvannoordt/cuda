# cuda
A repository for the implementation and benchmarking of a mandelbrot renderer

To install CUDA, first follow the install procedure here:
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

I got an issue when this happened, so I had to apply a little fix. See comment #9 here:
https://devtalk.nvidia.com/default/topic/1032886/cuda-setup-and-installation/unable-to-properly-install-uninstall-cuda-on-ubuntu-18-04/

Verify installation by running the following:  

>cat /proc/driver/nvidia/version

>>NVRM version: NVIDIA UNIX x86_64 Kernel Module  390.116  Sun Jan 27 07:21:36 PST 2019
  GCC version:  gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1)

and then running: 

>nvcc -V

>>nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2017 NVIDIA Corporation
  Built on Fri_Nov__3_21:07:56_CDT_2017
  Cuda compilation tools, release 9.1, V9.1.85
