#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <iomanip> 

// Constants defined by the MD5 algorithm
constexpr uint32_t A = 0x67452301;
constexpr uint32_t B = 0xefcdab89;
constexpr uint32_t C = 0x98badcfe;
constexpr uint32_t D = 0x10325476;

constexpr uint32_t S[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                          5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                          4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                          6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

constexpr uint32_t K[] = {0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
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
constexpr uint8_t PADDING[] = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
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
uint32_t rotateLeft(uint32_t x, uint32_t n) 
{
    return (x << n) | (x >> (32 - n));
}

// MD5Context struct
struct MD5Context 
{
    uint64_t size = 0;
    uint32_t buffer[4] = {A, B, C, D};
    uint8_t input[64] = {0};
    uint8_t digest[16] = {0};
};



// Step on 512 bits of input with the main MD5 algorithm.
void md5Step(uint32_t* buffer, uint32_t* input) 
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

// Initialize a context
void md5Init(MD5Context& ctx)
{
    ctx.size = 0;
    ctx.buffer[0] = A;
    ctx.buffer[1] = B;
    ctx.buffer[2] = C;
    ctx.buffer[3] = D;
}

// Add some amount of input to the context
void md5Update(MD5Context& ctx, const uint8_t* input_buffer, size_t input_len) {
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    ctx.size += static_cast<uint64_t>(input_len);

    // Copy each byte in input_buffer into the next space in our context input
    for (size_t i = 0; i < input_len; ++i) {
        ctx.input[offset++] = input_buffer[i];

        // If we've filled our context input, copy it into our local array input
        // then reset the offset to 0 and fill in a new buffer.
        // Every time we fill out a chunk, we run it through the algorithm
        // to enable some back and forth between CPU and I/O
        if (offset % 64 == 0) {
            for (size_t j = 0; j < 16; ++j) {
                // Convert to little-endian
                // The local variable `input` is our 512-bit chunk separated into 32-bit words
                // we can use in calculations
                input[j] = static_cast<uint32_t>(ctx.input[(j * 4) + 3]) << 24 |
                           static_cast<uint32_t>(ctx.input[(j * 4) + 2]) << 16 |
                           static_cast<uint32_t>(ctx.input[(j * 4) + 1]) << 8 |
                           static_cast<uint32_t>(ctx.input[(j * 4)]);
            }
            md5Step(ctx.buffer, input);
            offset = 0;
        }
    }
}

// Pad the current input and finalize the MD5 hash
void md5Finalize(MD5Context& ctx) {
    uint32_t input[16];
    unsigned int offset = static_cast<unsigned int>(ctx.size % 64);
    unsigned int padding_length = (offset < 56) ? (56 - offset) : (120 - offset);

    // Fill in the padding and undo the changes to size that resulted from the update
    md5Update(ctx, PADDING, padding_length);
    ctx.size -= static_cast<uint64_t>(padding_length);

    // Do a final update (internal to this function)
    // Last two 32-bit words are the two halves of the size (converted from bytes to bits)
    for (unsigned int j = 0; j < 14; ++j) {
        input[j] = static_cast<uint32_t>(ctx.input[(j * 4) + 3]) << 24 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 2]) << 16 |
                   static_cast<uint32_t>(ctx.input[(j * 4) + 1]) << 8 |
                   static_cast<uint32_t>(ctx.input[(j * 4)]);
    }
    input[14] = static_cast<uint32_t>(ctx.size * 8);
    input[15] = static_cast<uint32_t>(ctx.size >> 32);

    md5Step(ctx.buffer, input);

    // Move the result into digest (convert from little-endian)
    for (unsigned int i = 0; i < 4; ++i) {
        ctx.digest[(i * 4) + 0] = static_cast<uint8_t>(ctx.buffer[i] & 0x000000FF);
        ctx.digest[(i * 4) + 1] = static_cast<uint8_t>((ctx.buffer[i] & 0x0000FF00) >> 8);
        ctx.digest[(i * 4) + 2] = static_cast<uint8_t>((ctx.buffer[i] & 0x00FF0000) >> 16);
        ctx.digest[(i * 4) + 3] = static_cast<uint8_t>((ctx.buffer[i] & 0xFF000000) >> 24);
    }
}

// Run the MD5 algorithm on the provided input and store the digest in result
void md5String(const std::string& input, uint8_t* result)
{
    MD5Context ctx;
    md5Init(ctx);
    md5Update(ctx, reinterpret_cast<const uint8_t*>(input.c_str()), input.length());
    md5Finalize(ctx);
    std::memcpy(result, ctx.digest, 16);
}


int main() 
{
    
    std::string input_str = "Hello, MD5!";
    uint8_t digest[16];

    md5String(input_str, digest);

   std::ostringstream md5_result;
    for (int i = 0; i < 16; ++i) {
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
     std::string md5_hex_string = md5_result.str();
     std::cout << "MD5 Digest: " << md5_hex_string<< std::endl;
   

    return 0;
}
