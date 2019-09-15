#include <stdio.h>


int main() 
{
	int nDevices;
	// Gets properties of all installed NVIDIA GPUs.
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) 
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
		printf("  Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("  Max Threads Dim: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("  Global Memory (GB): %f\n\n", prop.totalGlobalMem / (1024.0f*1024.0f * 1024.0f));
	}
	return 0;
}
