//Same as the 2D example, but for 3D.

//Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	char* devPtr = devPitchedPtr.ptr;// There is an error on this line that I cannot figure out...
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch*height;
	for (int z = 0; z < depth; z++)
	{
		char* slice = devPtr + z*slicePitch;
		for (int y = 0; y < height; y++)
		{
			float* row = (float*)(slice + y*pitch);
			for (int x = 0; x < width; x++)
			{
				float element = row[x];
			}
		}
	}
}

//Host code
int main()
{
	int width = 64, height = 64, depth = 64;
	cudaExtent extent = make_cudaExtent(width*sizeof(float), height, depth);
	cudaPitchedPtr devPitchedPtr;
	cudaMalloc3D(&devPitchedPtr, extent);
	MyKernel<<<100,512>>>(devPitchedPtr, width, height, depth);
	return 0;
}
