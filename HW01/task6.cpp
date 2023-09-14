#include <iostream>
#include <stdlib.h>
#include <stdio.h>

int main(int argc , char *argv[])
{
    int N = std::atoi(argv[1]);
    std::cout<<"Getting N -command line argument from user: ";

    std::cout<<"\nPrinting the numbers in ascending from 0 to N \n";

    for(int i=0;i<=N;i++)
      {
        printf("%d ",i);
      }

    std::cout<<"\nPrinting the numbers in descending from N to 0 \n";

    for(int j=N;j>=0;j--)
      {
        std::cout<<j<<" ";
      }

    return 0;
}

