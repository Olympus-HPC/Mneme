/*************************************************************************
 *
 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 

 ====== This file has been modified by Xin Li on 2015-08-21 ======

 **************************************************************************/

#define MAXROOTS 7

double rys_contr_coulomb(int lena,double *aexps,double *acoefs,double *anorms,
                         double xa,double ya,double za,int *la,int *ma,int *na,
                         int lenb,double *bexps,double *bcoefs,double *bnorms,
                         double xb,double yb,double zb,int *lb,int *mb,int *nb,
                         int lenc,double *cexps,double *ccoefs,double *cnorms,
                         double xc,double yc,double zc,int *lc,int *mc,int *nc,
                         int lend,double *dexps,double *dcoefs,double *dnorms,
                         double xd,double yd,double zd,int *ld,int *md,int *nd);

double rys_coulomb_repulsion(double xa,double ya,double za,double norma,
                             int la,int ma,int na,double alphaa,
                             double xb,double yb,double zb,double normb,
                             int lb,int mb,int nb,double alphab,
                             double xc,double yc,double zc,double normc,
                             int lc,int mc,int nc,double alphac,
                             double xd,double yd,double zd,double normd,
                             int ld,int md,int nd,double alphad);

void Roots(int n, double X, double roots[], double weights[]);
void Root123(int n, double X, double roots[], double weights[]);
void Root4(double X, double roots[], double weights[]);
void Root5(double X, double roots[], double weights[]);
void Root6(int n,double X, double roots[], double weights[]);
double Int1d(double t,int ix,int jx,int kx, int lx,
             double xi,double xj, double xk,double xl,
             double alphai,double alphaj,double alphak,double alphal,
             double G[][MAXROOTS]);
