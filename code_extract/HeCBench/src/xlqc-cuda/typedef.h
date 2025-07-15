/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  typedef.h                                                      
 License:   BSD 3-Clause License

 This software is provided by the copyright holders and contributors "as is"
 and any express or implied warranties, including, but not limited to, the
 implied warranties of merchantability and fitness for a particular purpose are
 disclaimed. In no event shall the copyright holder or contributors be liable
 for any direct, indirect, incidental, special, exemplary, or consequential
 damages (including, but not limited to, procurement of substitute goods or
 services; loss of use, data, or profits; or business interruption) however
 caused and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of the use
 of this software, even if advised of the possibility of such damage.
 *****************************************************************************/

// Cartesian dimension
#define CART_DIM 3

#define BLOCKSIZE 8
#define SCREEN_THR 1.0e-16

// number of basis functions
#define N_S   1
#define N_SP  4
#define N_P   3
#define N_D   5
#define N_D_CART   6

// maximal number of DIIS error matrices
#define MAX_DIIS_DIM 6

// maximal length of string
#define MAX_STR_LEN 256

// PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CODATA 2014: 
// 1 Hartree = 27.21138602 eV
// 1 Bohr = 0.52917721067 Angstrom
//#define HARTREE2EV 27.21138602
//#define BOHR2ANGSTROM 0.52917721067

// CODATA 2006
#define BOHR2ANGSTROM 0.5291772086
#define HARTREE2EV   27.2113839

// vector
typedef struct {
    double x, y, z;
} Vec_R;

// atomic name, position and nuclear charge
typedef struct {
    int num;
    char **name;
    double **pos;
    int *nuc_chg;
} Atom;

// basis set
typedef struct {
    int num;
    double **expon, **coef, **norm;
    double *xbas, *ybas, *zbas;
    int *nprims;
    int **lx, **ly, **lz;
} Basis;

