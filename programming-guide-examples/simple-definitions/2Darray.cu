//This code allocates a widthxheight 2D array of floats and 
//shows how to loop over the array elements in device code.

//Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
	for (int r = 0; r < height; r++)
	{
		float* row = (float*)((char*)devPtr + r*pitch);
		for (int c = 0; c < width; c++)
		{
			float element = row[c];
		}
	}
}

//Host code
int main()
{
	int width = 64, height = 64;
	float* devPtr; //devPtr is a pointer.
	size_t pitch;
	cudaMallocPitch(&devPtr, &pitch, width*sizeof(float), height);
	MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
	return 0;
}

