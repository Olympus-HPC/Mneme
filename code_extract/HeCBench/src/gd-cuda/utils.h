#ifndef UTILS_hpp
#define UTILS_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>      // std::stringstream, std::stringbuf
#include <iterator>
#include <cmath>
#include <cassert>
#include <sys/time.h>

//  Copyright © 2019 Naga V Gudapati. All rights reserved.

// compressed row storage
struct Classification_Data_CRS{
    
    long nzmax;  //maximum number of entries
    int m;  //Number of rows
    int n;  //number of columns
    
    std::vector<float> values;
    std::vector<int> col_index;
    std::vector<int> row_ptr;
    
    std::vector<int> y_label;

    Classification_Data_CRS() = default;
    
    Classification_Data_CRS(int m, int n, long nzmax) {
        this->m = m;
        this->n = n;
        this->nzmax = nzmax;
        this->row_ptr = std::vector<int>((n + 1));
        this->col_index = std::vector<int>(nzmax);
        this->values = std::vector<float>(nzmax);
    }
    
};



//   We shall try to read the libsvm file line by line and convert it into a simple compressed
//   row storage matrix. 
void get_CRSM_from_svm(Classification_Data_CRS &M, const std::string &filename);


long long get_time();

#endif /* UTILS_hpp */

