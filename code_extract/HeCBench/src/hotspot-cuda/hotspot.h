#ifndef HOTSPOT_H
#define HOTSPOT_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef RD_WG_SIZE_0_0                                                            
        #define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
        #define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
        #define BLOCK_SIZE RD_WG_SIZE                                            
#else                                                                                    
        #define BLOCK_SIZE 16                                                            
#endif                                                                                   

#define STR_SIZE 256
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/* maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD	(3.0e6)
/* required precision in degrees */
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor */
#define FACTOR_CHIP	0.5

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

/* chip parameters */
const static float t_chip = 0.0005;
const static float chip_height = 0.016;
const static float chip_width = 0.016;
/* ambient temperature, assuming no package at all */
const static float amb_temp = 80.0;

void writeoutput(float *, int, int, char *);
void readinput(float *, int, int, char *);
void usage(int, char **);
void run(int, char **);

#endif
