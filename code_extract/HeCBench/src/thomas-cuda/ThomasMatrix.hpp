/**
 *
 *  @file ThomasMatrix.cu
 *
 *  @brief cuThomasBatch kernel implementaion.
 *
 *  cuThomasBatch is a software package provided by
 *  Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *  @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *  @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <sstream>


template<typename T>
void copyContainer(std::vector<T> &start,T* &end ){

    end = (T*) malloc(start.size()*sizeof start[0]);
    std::copy(start.begin(),start.end(),end);

}

struct ThomasMatrix {
    double *a;
    double *b; 
    double *d; 
    double *rhs;
    int batchCount;
    int M;
};

double fRand(double fMin, double fMax){

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);

}

ThomasMatrix loadThomasMatrixSyn(int size){

    ThomasMatrix tm;

    std::vector<double> u;
    std::vector<double> l;
    std::vector<double> d;
    std::vector<double> rhs;

    for (int i = 0; i < size; ++i)
    {
      
        u.push_back(fRand((double)-2.,(double)2.));
        l.push_back(fRand((double)-2.,(double)2.));
        d.push_back(fRand((double)5.,(double)10.));
        rhs.push_back(fRand((double)-2.,(double)2.));
      
    }

    copyContainer(u,tm.a);
    copyContainer(l,tm.b);
    copyContainer(d,tm.d);
    copyContainer(rhs,tm.rhs);
    tm.batchCount=1;
    tm.M=size;

    return tm;

}
