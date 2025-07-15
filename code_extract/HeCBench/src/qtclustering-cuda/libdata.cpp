#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <map>

#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include "libdata.h"

#define MAX_WIDTH  20.0
#define MAX_HEIGHT 20.0

using namespace std;


static inline float frand(void){
#ifdef _WIN32
    return (float)rand()/RAND_MAX;
#else
    return (float)random()/RAND_MAX;
#endif
}

// This function generates elements as points on a 2D Euclidean plane confined
// in a MAX_WIDTHxMAX_HEIGHT square (20x20 by default).  The elements are not
// uniformly distributed on the plane, but rather appear in clusters of random
// radius and cardinality. The maximum cardinality of a cluster is N/30 where
// N is the total number of data generated.
float *generate_synthetic_data(float **rslt_mtrx, int **indr_mtrx, int *max_degree, float threshold, int N, int matrix_type_mask){
    int count, bound, D=0;
    float *dist_mtrx, *points;
    int *index_mtrx;
    float threshold_sq, min_dim;

    // Create N points in a MAX_WIDTH x MAX_HEIGHT (20x20) space.
    points = (float *)malloc(2*N*sizeof(float));

    min_dim = MAX_WIDTH < MAX_HEIGHT ? MAX_WIDTH : MAX_HEIGHT;

    count = 0;
    while( count < N ){
        int group_cnt;
        float R, cntr_x, cntr_y;

        // Create "group_cnt" points within a circle of radious "R"
        // around center point "(cntr_x, cntr_y)"

        cntr_x = frand()*MAX_WIDTH;
        cntr_y = frand()*MAX_HEIGHT;
        R = frand()*min_dim/2;
#ifdef _WIN32
        group_cnt = rand()%(N/30);
#else
        group_cnt = random()%(N/30);
#endif
        // make sure we don't make more points than we need
        if( group_cnt > (N-count) ){
            group_cnt = N-count;
        }

        while( group_cnt > 0 ){
            float sign, r, x, y, dx, dy;
            sign = (frand()<0.5)?-1.0:1.0;
            r = frand()*R;         // 0 <= r <= R
            dx = (2.0*frand()-1.0)*r;  // -r < dx < r
            dy = sqrtf(r*r-dx*dx)*sign; // y = (r^2-dx^2)^0.5
            x = cntr_x+dx;
            if( x<0 || x>MAX_WIDTH)
                continue;
            y = cntr_y+dy;
            if( y<0 || y>MAX_HEIGHT)
                continue;

            points[2*count]   = x;
            points[2*count+1] = y;

            count++;
            group_cnt--;
        }
    }

    threshold_sq = threshold*threshold;

    // Allocate the proper size matrix
    for(int i=0; i<N; i++){
        int delta = 0;

        float p1_x = points[2*i];
        float p1_y = points[2*i+1];
        for(int j=0; j<N; j++){
            if( j == i ){
                continue;
            }
            float p2_x = points[2*j];
            float p2_y = points[2*j+1];
            float dist_sq = (p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y);
            if( dist_sq < threshold_sq ){
                delta++;
            }
        }
        if( delta > D )
            D = delta;
    }

        bound = D;
    dist_mtrx = (float*) malloc (sizeof(float)*N*D);
    index_mtrx = (int*) malloc (sizeof(int)*N*D);

    // Initialize the distances
    for(int i=0; i<N; i++){
        for(int j=0; j<D; j++){
            index_mtrx[i*D+j] = -1;
        }
        for(int j=0; j<bound; j++){
            dist_mtrx[i*bound+j] = FLT_MAX;
        }
    }

    for(int i=0; i<N; i++){
        int delta = 0;
        float p1_x, p1_y;

        p1_x = points[2*i];
        p1_y = points[2*i+1];
        for(int j=0; j<N; j++){ // This is supposed to be "N", not "Delta"
            float p2_x, p2_y, dist_sq;
            if( j == i ){
                continue;
            }
            p2_x = points[2*j];
            p2_y = points[2*j+1];
            dist_sq = (p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y);
            if( dist_sq < threshold_sq ){
                float dist = (float)sqrt((double)dist_sq);
                index_mtrx[i*D+delta] = j;
                    dist_mtrx[i*D+delta] = dist;
                delta++;
            }
        }
    }

    *max_degree = D;
    *rslt_mtrx = dist_mtrx;
    *indr_mtrx = index_mtrx;
    return points;
}
