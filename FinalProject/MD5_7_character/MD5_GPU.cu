#include <omp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <chrono>

#define tmax 62
#define idmax
#define numBlocks 923522
#define threadsPerBlock 992

using std::chrono::duration;

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
__device__ void md5Step(uint32_t* buffer, uint32_t* input)
{
    uint32_t AA = buffer[0];
    uint32_t BB = buffer[1];
    uint32_t CC = buffer[2];
    uint32_t DD = buffer[3];

    uint32_t E;

    unsigned int j;

    for(unsigned int i = 0; i < 64; ++i)
    {
        switch(i / 16)
	{
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

__device__ void md5Init(MD5_Struct& content)
{
    content.size = 0;
    content.buffer[0] = A;
    content.buffer[1] = B;
    content.buffer[2] = C;
    content.buffer[3] = D;
}

__device__ void md5Update(MD5_Struct& content, const uint8_t* input_buffer, size_t input_len)
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
        if (offset % 64 == 0) 
	{
    	    	
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
	   

            md5Step(content.buffer, input);
            offset = 0;
        }
    }
}


__device__ void md5Finalize(MD5_Struct& content) 
{
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(content.size % 64);
    unsigned int padding_length = (offset < 56) ? (56 - offset) : (120 - offset);

    // Fill in the padding and undo the changes to size that resulted from the update
    md5Update(content, PADDING, padding_length);
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

    md5Step(content.buffer, input);

    // Move the result into digest (convert from little-endian)
  
    /*
       for (unsigned int i = 0; i < 4; ++i)
    {
        content.digest[(i * 4) + 0] = static_cast<uint8_t>(content.buffer[i] & 0x000000FF);
        content.digest[(i * 4) + 1] = static_cast<uint8_t>((content.buffer[i] & 0x0000FF00) >> 8);
        content.digest[(i * 4) + 2] = static_cast<uint8_t>((content.buffer[i] & 0x00FF0000) >> 16);
        content.digest[(i * 4) + 3] = static_cast<uint8_t>((content.buffer[i] & 0xFF000000) >> 24);
    }
    */
    content.digest[(0  * 4) + 0] = static_cast<uint8_t>(content.buffer[0] & 0x000000FF);
        content.digest[(0  * 4) + 1] = static_cast<uint8_t>((content.buffer[0] & 0x0000FF00) >> 8);
        content.digest[(0  * 4) + 2] = static_cast<uint8_t>((content.buffer[0] & 0x00FF0000) >> 16);
        content.digest[(0  * 4) + 3] = static_cast<uint8_t>((content.buffer[0] & 0xFF000000) >> 24);
		
		content.digest[(1  * 4) + 0] = static_cast<uint8_t>(content.buffer[1] & 0x000000FF);
        content.digest[(1  * 4) + 1] = static_cast<uint8_t>((content.buffer[1] & 0x0000FF00) >> 8);
        content.digest[(1  * 4) + 2] = static_cast<uint8_t>((content.buffer[1] & 0x00FF0000) >> 16);
        content.digest[(1  * 4) + 3] = static_cast<uint8_t>((content.buffer[1] & 0xFF000000) >> 24);
		
		content.digest[(2  * 4) + 0] = static_cast<uint8_t>(content.buffer[2] & 0x000000FF);
        content.digest[(2  * 4) + 1] = static_cast<uint8_t>((content.buffer[2] & 0x0000FF00) >> 8);
        content.digest[(2  * 4) + 2] = static_cast<uint8_t>((content.buffer[2] & 0x00FF0000) >> 16);
        content.digest[(2  * 4) + 3] = static_cast<uint8_t>((content.buffer[2] & 0xFF000000) >> 24);
		
		content.digest[(3  * 4) + 0] = static_cast<uint8_t>(content.buffer[3] & 0x000000FF);
        content.digest[(3  * 4) + 1] = static_cast<uint8_t>((content.buffer[3] & 0x0000FF00) >> 8);
        content.digest[(3  * 4) + 2] = static_cast<uint8_t>((content.buffer[3] & 0x00FF0000) >> 16);
        content.digest[(3  * 4) + 3] = static_cast<uint8_t>((content.buffer[3] & 0xFF000000) >> 24);

}


__device__ void md5String(const char *input, uint8_t *result)
 {
    MD5_Struct content;
    md5Init(content);
    md5Update(content, reinterpret_cast<const uint8_t*>(input), 7); //len
    md5Finalize(content);
   
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



__device__ void formatDigestToHex(const unsigned char *digest, char *output) 
{
    const char* hexChars = "0123456789abcdef";
  
    /*
     for (int i = 0; i < 16; ++i)
    {
        output[i * 2] = hexChars[(digest[i] >> 4) & 0x0F];
        output[i * 2 + 1] = hexChars[digest[i] & 0x0F];
    }
    */
    output[0 * 2] = hexChars[(digest[0] >> 4) & 0x0F];
        output[0 * 2 + 1] = hexChars[digest[0] & 0x0F];
		
		output[1 * 2] = hexChars[(digest[1] >> 4) & 0x0F];
        output[1 * 2 + 1] = hexChars[digest[1] & 0x0F];
		
		output[2 * 2] = hexChars[(digest[2] >> 4) & 0x0F];
        output[2 * 2 + 1] = hexChars[digest[2] & 0x0F];
		
		output[3 * 2] = hexChars[(digest[3] >> 4) & 0x0F];
        output[3 * 2 + 1] = hexChars[digest[3] & 0x0F];
		
		output[4 * 2] = hexChars[(digest[4] >> 4) & 0x0F];
        output[4 * 2 + 1] = hexChars[digest[4] & 0x0F];
		
		output[5 * 2] = hexChars[(digest[5] >> 4) & 0x0F];
        output[5 * 2 + 1] = hexChars[digest[5] & 0x0F];
		
		output[6 * 2] = hexChars[(digest[6] >> 4) & 0x0F];
        output[6 * 2 + 1] = hexChars[digest[6] & 0x0F];
		
		output[7 * 2] = hexChars[(digest[7] >> 4) & 0x0F];
        output[7 * 2 + 1] = hexChars[digest[7] & 0x0F];
		
		output[8 * 2] = hexChars[(digest[8] >> 4) & 0x0F];
        output[8 * 2 + 1] = hexChars[digest[8] & 0x0F];
		
		output[9 * 2] = hexChars[(digest[9] >> 4) & 0x0F];
        output[9 * 2 + 1] = hexChars[digest[9] & 0x0F];
		
		output[10 * 2] = hexChars[(digest[10] >> 4) & 0x0F];
        output[10 * 2 + 1] = hexChars[digest[10] & 0x0F];
		
		output[11 * 2] = hexChars[(digest[11] >> 4) & 0x0F];
        output[11 * 2 + 1] = hexChars[digest[11] & 0x0F];
		
		output[12 * 2] = hexChars[(digest[12] >> 4) & 0x0F];
        output[12 * 2 + 1] = hexChars[digest[12] & 0x0F];
		
		output[13 * 2] = hexChars[(digest[13] >> 4) & 0x0F];
        output[13 * 2 + 1] = hexChars[digest[13] & 0x0F];
		
		output[14 * 2] = hexChars[(digest[14] >> 4) & 0x0F];
        output[14 * 2 + 1] = hexChars[digest[14] & 0x0F];
		
		output[15 * 2] = hexChars[(digest[15] >> 4) & 0x0F];
        output[15 * 2 + 1] = hexChars[digest[15] & 0x0F];
   
    output[32] = '\0'; // Null-terminate the string
}

__constant__ char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

__global__ void bruteForceKernel(char prefix1, char prefix2, char *targetHash,int *flag ) 
{
    unsigned char hash[16];  // Assuming a 128-bit hash (like MD5)

    char password[8]={0};
    /*
    __shared__  char sharedTargetHash[32];	
    
    int tid = threadIdx.x;
    if (tid < 62) 
    {
        sharedTargetHash[tid] = targetHash[tid];
    }
    */
   

    password[0]=prefix1;
    password[1]=prefix2;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int remainingIdx = idx;
	password[2] = charset[remainingIdx % tmax];
        remainingIdx /= tmax;
	password[3] = charset[remainingIdx % tmax];
        remainingIdx /= tmax;
	password[4] = charset[remainingIdx % tmax];
        remainingIdx /= tmax;
	password[5] = charset[remainingIdx % tmax];
        remainingIdx /= tmax;
	password[6] = charset[remainingIdx % tmax];
        remainingIdx /= tmax;
	
	password[7]='\0';
	
	 char genhash[33];
    // Compute hash

        md5String(password,hash);
        formatDigestToHex(hash,genhash);


        
	
     __shared__ bool match;
     match=true;
    
    for(int i=0;i<32;++i)
    {
       if (genhash[i] != targetHash[i])
         {
            match = false;
            break;
         }
    }
  

   if (match) 
    {
	printf ("Password  cracked SMARTASSSS :%s\n",password);
	atomicExch(flag,1);
        return ;
        	
    }
}	

__global__ void md5Kernel(const char *Input, char *targetHash)
{
	unsigned char digest[16];
        md5String(Input, digest);
        formatDigestToHex(digest, targetHash); // Convert digest to hex string
}


__host__ void cudaBruteForce(char *h_targetHash)

{

    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    char *dev_targetHash;


    cudaMallocManaged((void**)&dev_targetHash, 33* sizeof(char));
    cudaMemcpy(dev_targetHash, h_targetHash, 33* sizeof( char) , cudaMemcpyHostToDevice);

    int *dev_flag;
    cudaMallocManaged((void**)&dev_flag, sizeof(int));


    *dev_flag = 0; // Initialize the flag on the device side

   
    auto start = std::chrono::high_resolution_clock::now();

        for(int i=0;i< 62;++i)
        {
          for(int j=0;j< 62;++j)
                {
                     bruteForceKernel<<<numBlocks, threadsPerBlock>>>(charset[i], charset[j], dev_targetHash, dev_flag);
		     cudaDeviceSynchronize();
		     if (*dev_flag){ break;}
                }

	     if (*dev_flag){ break;}
        }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Password cracked in: " << duration.count() << " milliseconds" << std::endl;

    char output[33];
    cudaMemcpy(output,dev_targetHash, 33* sizeof(char) , cudaMemcpyDeviceToHost);
    
    cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess)
            {
              std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
	    }

   cudaFree(dev_targetHash); 
   cudaFree(dev_flag);
}


int main(int argc, char *argv[]) 
{

    const char* Input = argv[1]; 
    char* d_input;
    char* d_targetHash;
    char h_targetHash[33];


    std::cout <<" Space Patrol Delta \n **************\n Enter your Bank Password: *******\n\n"; 
    cudaMallocManaged(&d_input, 8);  //len+1
    cudaMallocManaged(&d_targetHash, 33* sizeof(char) );
    cudaMemcpy(d_input, Input, 8, cudaMemcpyHostToDevice); //len+1


    md5Kernel<<<1,1>>>(d_input, d_targetHash);


    cudaMemcpy(h_targetHash, d_targetHash, 33* sizeof(char) , cudaMemcpyDeviceToHost);

    cudaBruteForce(h_targetHash);
    
    
    cudaFree(d_input);
    cudaFree(d_targetHash);
    return ;
}


