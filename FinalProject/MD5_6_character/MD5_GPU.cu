

int main(int argc, char *argv[]) 
{
    const char* Input = argv[1];
    char* d_input;
    char* d_targetHash;
    char h_targetHash[32];
    char output[32];


    cudaMalloc(&d_input, 7);
    cudaMalloc(&d_targetHash, 32 * sizeof(char));
    cudaMemcpy(d_input, Input, 7, cudaMemcpyHostToDevice);

    // Launch the kernel
    md5Kernel<<<1, 1>>>(d_input, d_targetHash);

    cudaMemcpy(h_targetHash, d_targetHash, 33 * sizeof(char), cudaMemcpyDeviceToHost);

    std::cout <<" Space Patrol Delta \n **************\n Enter your Bank Password: *******\n\n"; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char* dev_targetHash;
    cudaMalloc(&dev_targetHash, 32 * sizeof(char));
    cudaMemcpy(dev_targetHash, h_targetHash, 32 * sizeof(char), cudaMemcpyHostToDevice);

    //number of blocks = (totalCombinations + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start, 0);

    bruteForceKernel<<<numBlocks, threadsPerBlock>>>(dev_targetHash);
	
    cudaMemcpy(output,dev_targetHash,33*sizeof(char),cudaMemcpyDeviceToHost);	
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);


    // Calculate the elapsed time between start and stop
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "\nTotal time taken to crack the password :" << milliseconds << " milliseconds " <<std::endl;
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_targetHash);
    cudaFree(dev_targetHash);
   
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}


