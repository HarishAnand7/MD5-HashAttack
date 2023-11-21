#include <chrono>
#include <cstring>
#include <random>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include "optimize.h"
#include <iostream>
using namespace std;
using namespace chrono;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char *argv[])
 {
    int n = atoi(argv[1]);
    data_t *value = new data_t[n];
    vec v =vec(n);


    for(int i=0;i<n;i++)
    {
	    value[i]=1;
    }

    v.data=value;
    data_t store;
    double T = 0.0;

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;

    //Optimization 1 
 
    for(int iter =0;iter<10;iter++)
    {
    start = high_resolution_clock::now();
    optimize1(&v, &store);
    end = high_resolution_clock::now();
    T += duration_cast<duration<double, std::milli>>(end - start).count();
    }
    T/=10;
    cout << store << "\n";
    cout << T << endl;
    T = 0.0;

    //Optimization 2

    for(int iter =0;iter<10;iter++)
    {
    start = high_resolution_clock::now();
    optimize2(&v, &store);
    end = high_resolution_clock::now();
    T += duration_cast<duration<double, std::milli>>(end - start).count();
    }
    T/=10;
    cout << store << "\n";;
    cout << T << endl;

    T = 0.0;

    //Optimization 3

    for(int iter =0;iter<10;iter++)
    {
    start = high_resolution_clock::now();
    optimize3(&v, &store);
    end = high_resolution_clock::now();
    T += duration_cast<duration<double, std::milli>>(end - start).count();
    }
    T/=10;

    cout << store <<  "\n";;
    cout << T << endl;

    T = 0.0;

    //Optimization 4

    for(int iter =0;iter<10;iter++)
    {
    start = high_resolution_clock::now();
    optimize4(&v, &store);
    end = high_resolution_clock::now();
    T += duration_cast<duration<double, std::milli>>(end - start).count();
    }
    T/=10;

    cout << store <<  "\n";;
    cout << T << endl;

    T = 0.0;

    //Optimization 5

    for(int iter =0;iter<10;iter++)
    {
    start = high_resolution_clock::now();
    optimize5(&v, &store);
    end = high_resolution_clock::now();
    T += duration_cast<duration<double, std::milli>>(end - start).count();
    }
    T/=10;

    cout << store <<  "\n";;
    cout << T << endl;

    cout<<endl;


    }

    


