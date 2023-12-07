/ Function to perform a brute-force attack
void bruteForce(const std::string& targetHash)
 {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int passwordLength = 7;

    char password[8] = {0};  // Null-terminated string to hold the password

    int charsetSize = strlen(charset);

    for (int i = 0; i < charsetSize; ++i) 
	{
        password[0] = charset[i];

        for (int j = 0; j < charsetSize; ++j) 
	{
            password[1] = charset[j];

            for (int k = 0; k < charsetSize; ++k) 
	    {
                password[2] = charset[k];

                for (int l = 0; l < charsetSize; ++l)
		{
                    password[3] = charset[l];

                    for (int m = 0; m < charsetSize; ++m) 
		    {
                        password[4] = charset[m];

                        for (int n = 0; n < charsetSize; ++n) 	
			{
                            password[5] = charset[n];

                            for (int o = 0; o < charsetSize; ++o)
			    {
                                password[6] = charset[o];
                                std::cout<<password<<"\n";
                                std::string currentHash = md5String(password);
                                std::cout<<currentHash<<"\n";

                                if (currentHash == targetHash)
				{
                                    std::cout << "Password Cracked: " << password << std::endl;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "Password not found." << std::endl;
}
