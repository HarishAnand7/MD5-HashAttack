#include <iostream>
int main()
{
    int N;
    std::cout<<"Getting N -command line argument from user: ";
    std::cin>>N;
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

