//Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main()
{
	//stuff here
	//kernel invocation with N threads
	VecAdd<<<1, N>>>(A, B, C)
	//more stuff here
}
