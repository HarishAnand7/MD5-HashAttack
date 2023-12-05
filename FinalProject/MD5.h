void md5Step(uint32_t* buffer, uint32_t* input);

void md5Init(MD5Context& ctx);

void md5Update(MD5Context& ctx, const uint8_t* input_buffer, size_t input_len);

void md5Finalize(MD5Context& ctx);

void md5String(const std::string& input, uint8_t* result);

void md5File(const std::string& filepath, uint8_t* result);

