void bruteForce(std::string& targetHash, uint8_t* digest)
{
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int passwordLength = 5;
    char password[8] = {0};  // Null-terminated string to hold the password

    int charsetSize = strlen(charset);
    uint64_t totalCombinations = 1;


    // Calculate the total number of combinations
   totalCombinations = charsetSize * charsetSize * charsetSize * charsetSize * charsetSize;

   std::cout << "Total Combinations: " << totalCombinations << std::endl;

    for (uint64_t iteration = 0; iteration < totalCombinations; ++iteration)
      {
        // Generate the password using iteration as an index
        uint64_t temp = iteration;
        for (int i = 0; i < passwordLength; ++i)
        {
            password[i] = charset[temp % charsetSize];
            temp /= charsetSize;
        }

        md5String(password, digest);

        std::ostringstream md5_result;

        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[0]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[1]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[2]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[3]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[4]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[5]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[6]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[7]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[8]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[9]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[10]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[11]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[12]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[13]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[14]);
        md5_result << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[15]);


        std::string md5_hex_string = md5_result.str();

        //std::cout << "Iteration: " << iteration << ", Password: " << password << ", MD5: " << md5_hex_string << std::endl;

        if (md5_hex_string == targetHash)
        {
            std::cout << "Password Cracked: " << password << std::endl << md5_hex_string << ":\t" << iteration << std::endl;
            return;
        }
    }

    std::cout << "Password not found." << std::endl;
}
