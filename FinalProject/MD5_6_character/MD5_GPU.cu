#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cuda.h>
#define tmax 62
#define totalCombinations 56800235584 
#define numBlocks 57258303
#define threadsPerBlock  992

__constant__ uint32_t A = 0x67452301;
__constant__ uint32_t B = 0xefcdab89;
__constant__ uint32_t C = 0x98badcfe;
__constant__ uint32_t D = 0x10325476;

__constant__ uint32_t S[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                          5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                          4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                          6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

__constant__ uint32_t K[] = {0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
                       0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
                       0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
                       0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
                       0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
                       0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
                       0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
                       0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
                       0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
                       0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
                       0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
                       0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
                       0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
                       0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
                       0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
                       0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

// Padding used to make the size (in bits) of the input congruent to 448 mod 512
__constant__ uint8_t PADDING[] = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};


// Bit-manipulation functions defined by the MD5 algorithm
#define F(X, Y, Z) ((X & Y) | (~X & Z))
#define G(X, Y, Z) ((X & Z) | (Y & ~Z))
#define H(X, Y, Z) (X ^ Y ^ Z)
#define I(X, Y, Z) (Y ^ (X | ~Z))

// Rotates a 32-bit word left by n bits
__device__ uint32_t rotateLeft(uint32_t x, uint32_t n) 
{
    return (x << n) | (x >> (32 - n));
}
// MD5_Struct struct
struct MD5_Struct 
{
    uint64_t size = 0;
    uint32_t buffer[4] = {A, B, C, D};
    uint8_t input[64] = {0};
    uint8_t digest[16] = {0};
};


// Step on 512 bits of input with the main MD5 algorithm.
__device__ void MD5Step(uint32_t* buffer, uint32_t* input)
{
    uint32_t AA = buffer[0];
    uint32_t BB = buffer[1];
    uint32_t CC = buffer[2];
    uint32_t DD = buffer[3];

    uint32_t E;

    unsigned int j;

    for(unsigned int i = 0; i < 64; ++i){
        switch(i / 16){
            case 0:
                E = F(BB, CC, DD);
                j = i;
                break;
            case 1:
                E = G(BB, CC, DD);
                j = ((i * 5) + 1) % 16;
                break;
            case 2:
                E = H(BB, CC, DD);
                j = ((i * 3) + 5) % 16;
                break;
            default:
                E = I(BB, CC, DD);
                j = (i * 7) % 16;
                break;
        }

        uint32_t temp = DD;
        DD = CC;
        CC = BB;
        BB = BB + rotateLeft(AA + E + K[i] + input[j], S[i]);
        AA = temp;
    }

    buffer[0] += AA;
    buffer[1] += BB;
    buffer[2] += CC;
    buffer[3] += DD;
}

__device__ void MD5Init(MD5_Struct& content)
{
    content.size = 0;
    content.buffer[0] = A;
    content.buffer[1] = B;
    content.buffer[2] = C;
    content.buffer[3] = D;
}

__device__ void MD5Update(MD5_Struct& content, const uint8_t* input_buffer, size_t input_len)
{
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(content.size % 64);
    content.size += static_cast<uint64_t>(input_len);

    // Copy each byte in input_buffer into the next space in our context input
    for (size_t i = 0; i < input_len; ++i) 
    {
        content.input[offset++] = input_buffer[i];

        // If we've filled our context input, copy it into our local array input
        // then reset the offset to 0 and fill in a new buffer.
        // Every time we fill out a chunk, we run it through the algorithm
        // to enable some back and forth between CPU and I/O
        if (offset % 64 == 0) {
            for (size_t j = 0; j < 16; ++j) 
	    {
                // Convert to little-endian
                // The local variable `input` is our 512-bit chunk separated into 32-bit words
                // we can use in calculations
                input[j] = static_cast<uint32_t>(content.input[(j * 4) + 3]) << 24 |
                           static_cast<uint32_t>(content.input[(j * 4) + 2]) << 16 |
                           static_cast<uint32_t>(content.input[(j * 4) + 1]) << 8 |
                           static_cast<uint32_t>(content.input[(j * 4)]);
            }
            MD5Step(content.buffer, input);
            offset = 0;
        }
    }
}

__device__ void MD5Finalize(MD5_Struct& content) 
{
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(content.size % 64);
    unsigned int padding_length = (offset < 56) ? (56 - offset) : (120 - offset);

    // Fill in the padding and undo the changes to size that resulted from the update
    MD5Update(content, PADDING, padding_length);
    content.size -= static_cast<uint64_t>(padding_length);

    // Do a final update (internal to this function)
    // Last two 32-bit words are the two halves of the size (converted from bytes to bits)
    for (unsigned int j = 0; j < 14; ++j) 
    {
        input[j] = static_cast<uint32_t>(content.input[(j * 4) + 3]) << 24 |
                   static_cast<uint32_t>(content.input[(j * 4) + 2]) << 16 |
                   static_cast<uint32_t>(content.input[(j * 4) + 1]) << 8 |
                   static_cast<uint32_t>(content.input[(j * 4)]);
    }
    input[14] = static_cast<uint32_t>(content.size * 8);
    input[15] = static_cast<uint32_t>(content.size >> 32);

    MD5Step(content.buffer, input);

    // Move the result into digest (convert from little-endian)
    for (unsigned int i = 0; i < 4; ++i) {
        content.digest[(i * 4) + 0] = static_cast<uint8_t>(content.buffer[i] & 0x000000FF);
        content.digest[(i * 4) + 1] = static_cast<uint8_t>((content.buffer[i] & 0x0000FF00) >> 8);
        content.digest[(i * 4) + 2] = static_cast<uint8_t>((content.buffer[i] & 0x00FF0000) >> 16);
        content.digest[(i * 4) + 3] = static_cast<uint8_t>((content.buffer[i] & 0xFF000000) >> 24);
    }
}

__device__ void MD5String(const char *input, uint8_t *result)
 {
    MD5_Struct content;
    MD5Init(content);
    MD5Update(content, reinterpret_cast<const uint8_t*>(input), 6);
    MD5Finalize(content);
   
          result[0] = content.digest[0];
 	  result[1] = content.digest[1];
          result[2] = content.digest[2];
	  result[3] = content.digest[3];
          result[4] = content.digest[4];
	  result[5] = content.digest[5];
          result[6] = content.digest[6];
	  result[7] = content.digest[7];
          result[8] = content.digest[8];
	  result[9] = content.digest[9];
          result[10] = content.digest[10];
	  result[11] = content.digest[11];
          result[12] = content.digest[12];
	  result[13] = content.digest[13];
	  result[14] = content.digest[14];
	  result[15] = content.digest[15];
      
    
}
__device__ void FormatDigestToHex(const unsigned char *digest, char *output) 
{
    const char* hexChars = "0123456789abcdef";

    for (int i = 0; i < 16; ++i) 
    {
        output[i * 2] = hexChars[(digest[i] >> 4) & 0x0F];
        output[i * 2 + 1] = hexChars[digest[i] & 0x0F];
    }
    output[32] = '\0'; // Null-terminate the string
}


__global__ void MD5Kernel(const char *Input, char *targetHash) 
{
    unsigned char digest[16];
    MD5String(Input, digest);  // Assuming md5String computes the MD5 hash

    FormatDigestToHex(digest, targetHash);  // Convert digest to hex string
}

__constant__ char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
__global__ void bruteForceKernel(char *targetHash) 
{
    char password[7] = {0};  // 6 characters + null terminator
    unsigned char hash[16];  

    long long idx =(long long)  blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= totalCombinations) return;

    // Compute the n-th combination
    long long tmpIdx = idx;
    password[0] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;
    password[1] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;
    password[2] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;
    password[3] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;
    password[4] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;
    password[5] = charset[tmpIdx % tmax];
    tmpIdx /= tmax;


    password[6]='\0';

    __shared__ bool match;
    match=true;


    char genhash[33];
    MD5String(password,hash);
    FormatDigestToHex(hash,genhash);

      if ((genhash[0] != targetHash[0])|(genhash[1] != targetHash[1])|(genhash[2] != targetHash[2])|(genhash[3] != targetHash[3])
  |(genhash[4] != targetHash[4]) |(genhash[5] != targetHash[5]) |(genhash[6] != targetHash[6]) | (genhash[7] != targetHash[7])
  | (genhash[8] != targetHash[8])| (genhash[9] != targetHash[9])  | (genhash[10] != targetHash[10])| (genhash[11] != targetHash[11])
  | (genhash[12] != targetHash[12])  | (genhash[13] != targetHash[13]) | (genhash[14] != targetHash[14]) | (genhash[15] != targetHash[15])
  | (genhash[16] != targetHash[16]) | (genhash[17] != targetHash[17])  | (genhash[18] != targetHash[18])| (genhash[19] != targetHash[19])
  | (genhash[20] != targetHash[20]) | (genhash[21] != targetHash[21]) | (genhash[22] != targetHash[22])  | (genhash[23] != targetHash[23])
  | (genhash[24] != targetHash[24]) | (genhash[25] != targetHash[25])| (genhash[26] != targetHash[26])  | (genhash[27] != targetHash[27])
  | (genhash[28] != targetHash[28]) | (genhash[39] != targetHash[29]) | (genhash[30] != targetHash[30]) | (genhash[31] != targetHash[31]))
      {  match =false;}


    if (match) 
    {
        printf("Password Cracked SMARTASSS !!!! :%s\n", password);
        return;
        	
    }
	
    

}




int main(int argc, char *argv[]) 
{
    const char* Input = argv[1];
    char* d_input;
    char* d_targetHash;
    char h_targetHash[33];
    char output[33];


    cudaMallocManaged(&d_input, 7);
    cudaMallocManaged(&d_targetHash, 33 * sizeof(char));
    cudaMemcpy(d_input, Input, 7, cudaMemcpyHostToDevice);

    // Launch the kernel
    MD5Kernel<<<1, 1>>>(d_input, d_targetHash);

    cudaMemcpy(h_targetHash, d_targetHash, 33 * sizeof(char), cudaMemcpyDeviceToHost);

    std::cout <<" Space Patrol Delta \n **************\n Enter your Bank Password: *******\n\n"; 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char* dev_targetHash;
    cudaMallocManaged(&dev_targetHash, 33 * sizeof(char));
    cudaMemcpy(dev_targetHash, h_targetHash, 33 * sizeof(char), cudaMemcpyHostToDevice);

    //number of blocks = (totalCombinations + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start, 0);

    bruteForceKernel<<<numBlocks, threadsPerBlock>>>(dev_targetHash);
	
    cudaMemcpy(output,dev_targetHash,33 * sizeof(char),cudaMemcpyDeviceToHost);	
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



