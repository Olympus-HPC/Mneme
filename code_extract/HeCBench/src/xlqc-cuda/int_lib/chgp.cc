/**********************************************************************
 * chgp.c  C implementation of two-electron integral code in hgp.py
 *
 * Implementation of Head-Gordon & Pople's scheme for electron repulsion
 *  integrals (ref), which, in turn, derives from Saika and Obarra's scheme.
 *
 * Routines:
 * hrr performs the horizontal recursion relationships
 * vrr performs the vertical recursion relationship

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 

 ====== This file has been modified by Xin Li on 2015-08-17 ======

 **********************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cints.h"
#include "chgp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// sqrt(2.)*pow(M_PI,1.25)
#define SQRT_2_PI_1_25 5.91496717279561323721

double contr_hrr(int lena, double xa, double ya, double za, double *anorms,
		 int la, int ma, int na, double *aexps, double *acoefs,
		 int lenb, double xb, double yb, double zb, double *bnorms,
		 int lb, int mb, int nb, double *bexps, double *bcoefs,
		 int lenc, double xc, double yc, double zc, double *cnorms,
		 int lc, int mc, int nc, double *cexps, double *ccoefs,
		 int lend, double xd, double yd, double zd, double *dnorms,
		 int ld, int md, int nd, double *dexps, double *dcoefs)
{
  if (lb > 0) 
  {
    return contr_hrr(lena,xa,ya,za,anorms,la+1,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb-1,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (xa-xb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb-1,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }
  else if (mb > 0)
  {
    return contr_hrr(lena,xa,ya,za,anorms,la,ma+1,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb-1,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (ya-yb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb-1,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }
  else if (nb > 0)
  {
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na+1,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb-1,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs)
      + (za-zb)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb-1,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd,dexps,dcoefs);
  }
  else if (ld > 0)
  {
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc+1,mc,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld-1,md,nd,dexps,dcoefs)
      + (xc-xd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld-1,md,nd,dexps,dcoefs);
  }
  else if (md > 0)
  {
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc+1,nc,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md-1,nd,dexps,dcoefs)
      + (yc-yd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md-1,nd,dexps,dcoefs);
  }
  else if (nd > 0)
  {
    return contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
		     lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
		     lenc,xc,yc,zc,cnorms,lc,mc,nc+1,cexps,ccoefs,
		     lend,xd,yd,zd,dnorms,ld,md,nd-1,dexps,dcoefs)
      + (zc-zd)*contr_hrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
			  lenb,xb,yb,zb,bnorms,lb,mb,nb,bexps,bcoefs,
			  lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
			  lend,xd,yd,zd,dnorms,ld,md,nd-1,dexps,dcoefs);
  }
  else
  {
    return contr_vrr(lena,xa,ya,za,anorms,la,ma,na,aexps,acoefs,
             lenb,xb,yb,zb,bnorms,bexps,bcoefs,
             lenc,xc,yc,zc,cnorms,lc,mc,nc,cexps,ccoefs,
             lend,xd,yd,zd,dnorms,dexps,dcoefs);
  }
}

double contr_vrr(int lena, double xa, double ya, double za,
			double *anorms, int la, int ma, int na,
			double *aexps, double *acoefs,
			int lenb, double xb, double yb, double zb,
			double *bnorms, double *bexps, double *bcoefs,
			int lenc, double xc, double yc, double zc,
			double *cnorms, int lc, int mc, int nc,
			double *cexps, double *ccoefs,
			int lend, double xd, double yd, double zd,
			double *dnorms, double *dexps, double *dcoefs)
{
  int i,j,k,l;
  double val=0.;
  for (i=0; i<lena; ++i)
  {
    for (j=0; j<lenb; ++j)
	{
      for (k=0; k<lenc; ++k)
	  {
        for (l=0; l<lend; ++l)
		{
          val += acoefs[i]*bcoefs[j]*ccoefs[k]*dcoefs[l]*
                 vrr(xa,ya,za,anorms[i],la,ma,na,aexps[i],
                     xb,yb,zb,bnorms[j],bexps[j],
                     xc,yc,zc,cnorms[k],lc,mc,nc,cexps[k],
                     xd,yd,zd,dnorms[l],dexps[l],0);
		}
	  }
	}
  }
  return val;
}


double vrr(double xa, double ya, double za, double norma,
           int la, int ma, int na, double alphaa,
           double xb, double yb, double zb, double normb, double alphab,
           double xc, double yc, double zc, double normc,
           int lc, int mc, int nc, double alphac,
           double xd, double yd, double zd, double normd, double alphad,
           int m)
{
  int *ac_lmax_prod = (int *)malloc(sizeof(int) * 6);
  ac_lmax_prod[0] = la + 1;
  ac_lmax_prod[1] = ac_lmax_prod[0] * (ma + 1);
  ac_lmax_prod[2] = ac_lmax_prod[1] * (na + 1);
  ac_lmax_prod[3] = ac_lmax_prod[2] * (lc + 1);
  ac_lmax_prod[4] = ac_lmax_prod[3] * (mc + 1);
  ac_lmax_prod[5] = ac_lmax_prod[4] * (nc + 1);

  double px,py,pz,qx,qy,qz,zeta,eta,wx,wy,wz,rab2,rcd2,Kcd,rpq2,T,Kab,val;

  int i,j,k,q,r,s,im,mtot;

  px = product_center_1D(alphaa,xa,alphab,xb);
  py = product_center_1D(alphaa,ya,alphab,yb);
  pz = product_center_1D(alphaa,za,alphab,zb);

  qx = product_center_1D(alphac,xc,alphad,xd);
  qy = product_center_1D(alphac,yc,alphad,yd);
  qz = product_center_1D(alphac,zc,alphad,zd);

  zeta = alphaa+alphab;
  eta = alphac+alphad;

  double ZE = zeta + eta;
  double SQRT_ZE = sqrt(ZE);
  double inv_ZE = 1.0 / ZE;
  double E_ZE = eta * inv_ZE;
  double Z_ZE = zeta * inv_ZE;

  wx = product_center_1D(zeta,px,eta,qx);
  wy = product_center_1D(zeta,py,eta,qy);
  wz = product_center_1D(zeta,pz,eta,qz);

  rab2 = dist2(xa,ya,za,xb,yb,zb);
  Kab = SQRT_2_PI_1_25 / (alphaa+alphab)
        *exp(-alphaa*alphab/(alphaa+alphab)*rab2);
  rcd2 = dist2(xc,yc,zc,xd,yd,zd);
  Kcd = SQRT_2_PI_1_25 / (alphac+alphad)
        *exp(-alphac*alphad/(alphac+alphad)*rcd2);
  rpq2 = dist2(px,py,pz,qx,qy,qz);
  T = zeta* E_ZE *rpq2;

  mtot = la+ma+na+lc+mc+nc+m;

  double maxterms = ac_lmax_prod[5] * (mtot + 1);

  double *Fgterms = (double *)malloc(sizeof(double) * (mtot + 1));
  double *vrr_terms = (double *)malloc(sizeof(double) * maxterms);

  Fgterms[mtot] = Fgamma(mtot,T);

  for (im = mtot-1; im >= 0; -- im)
  {
    Fgterms[im]=(2.*T*Fgterms[im+1]+exp(-T))/(2.*im+1);
  }

  for (im = 0; im < mtot+1; ++ im)
  {
    vrr_terms[iindex(0,0,0, 0,0,0, im, ac_lmax_prod)] = 
      norma*normb*normc*normd*Kab*Kcd/ SQRT_ZE *Fgterms[im];
  }

  free(Fgterms);

  for (i=0; i<la; ++i)
  {
    for (im = 0; im < mtot-i; ++ im) 
	{
      vrr_terms[iindex(i+1,0,0, 0,0,0, im, ac_lmax_prod)] = 
        (px-xa) * vrr_terms[iindex(i,0,0, 0,0,0, im, ac_lmax_prod)]
        + (wx-px) * vrr_terms[iindex(i,0,0, 0,0,0, im+1, ac_lmax_prod)];
      
      if (i > 0)
	  {
        vrr_terms[iindex(i+1,0,0, 0,0,0, im, ac_lmax_prod)] += 
          i/2./zeta*(vrr_terms[iindex(i-1,0,0, 0,0,0, im, ac_lmax_prod)]
                     - E_ZE * vrr_terms[iindex(i-1,0,0, 0,0,0, im+1, ac_lmax_prod)]);
	  }
    }
  }  

  for (j=0; j<ma; ++j)
  {
    for (i=0; i<la+1; ++i)
	{
      for (im=0; im<mtot-i-j; ++im)
	  {
        vrr_terms[iindex(i,j+1,0, 0,0,0, im, ac_lmax_prod)] = 
          (py-ya)*vrr_terms[iindex(i,j,0, 0,0,0, im, ac_lmax_prod)]
          + (wy-py)*vrr_terms[iindex(i,j,0, 0,0,0, im+1, ac_lmax_prod)];

        if (j>0)
		{
          vrr_terms[iindex(i,j+1,0, 0,0,0, im, ac_lmax_prod)] +=
            j/2./zeta*(vrr_terms[iindex(i,j-1,0, 0,0,0, im, ac_lmax_prod)]
            - E_ZE * vrr_terms[iindex(i,j-1,0, 0,0,0, im+1, ac_lmax_prod)]);
		}
      }
    }
  }

  for (k=0; k<na; ++k)
  {
    for (j=0; j<ma+1; ++j)
	{
      for (i=0; i<la+1; ++i)
	  {
        for (im=0; im<mtot-i-j-k; ++im)
		{
          vrr_terms[iindex(i,j,k+1, 0,0,0, im, ac_lmax_prod)] = 
            (pz-za)*vrr_terms[iindex(i,j,k, 0,0,0, im, ac_lmax_prod)]
            + (wz-pz)*vrr_terms[iindex(i,j,k, 0,0,0, im+1, ac_lmax_prod)];

          if (k>0)
		  {
            vrr_terms[iindex(i,j,k+1, 0,0,0, im, ac_lmax_prod)] += 
              k/2./zeta*(vrr_terms[iindex(i,j,k-1, 0,0,0, im, ac_lmax_prod)]
                         - E_ZE * vrr_terms[iindex(i,j,k-1, 0,0,0, im+1, ac_lmax_prod)]);
		  }
        }
      }
    }
  }

  for (q=0; q<lc; ++q)
  {
    for (k=0; k<na+1; ++k)
	{
      for (j=0; j<ma+1; ++j)
	  {
        for (i=0; i<la+1; ++i)
		{
          for (im=0; im<mtot-i-j-k-q; ++im)
		  {
            vrr_terms[iindex(i,j,k, q+1,0,0, im, ac_lmax_prod)] = 
              (qx-xc)*vrr_terms[iindex(i,j,k, q,0,0, im, ac_lmax_prod)]
              + (wx-qx)*vrr_terms[iindex(i,j,k, q,0,0, im+1, ac_lmax_prod)];

            if (q>0)
			{
              vrr_terms[iindex(i,j,k, q+1,0,0, im, ac_lmax_prod)] += 
                q/2./eta*(vrr_terms[iindex(i,j,k, q-1,0,0, im, ac_lmax_prod)]
             	          - Z_ZE * vrr_terms[iindex(i,j,k, q-1,0,0, im+1, ac_lmax_prod)]);
			}

            if (i>0)
			{
              vrr_terms[iindex(i,j,k, q+1,0,0, im, ac_lmax_prod)] += 
                i/2.* inv_ZE *vrr_terms[iindex(i-1,j,k, q,0,0, im+1, ac_lmax_prod)];
			}
          }
        }
      }
    }
  }

  for (r=0; r<mc; ++r)
  {
    for (q=0; q<lc+1; ++q)
	{
      for (k=0; k<na+1; ++k)
	  {
        for (j=0; j<ma+1; ++j)
		{
          for (i=0; i<la+1; ++i)
		  {
            for (im=0; im<mtot-i-j-k-q-r; ++im)
			{
              vrr_terms[iindex(i,j,k, q,r+1,0, im, ac_lmax_prod)] = 
                (qy-yc)*vrr_terms[iindex(i,j,k, q,r,0, im, ac_lmax_prod)]
                + (wy-qy)*vrr_terms[iindex(i,j,k, q,r,0, im+1, ac_lmax_prod)];

              if (r>0)
			  {
                vrr_terms[iindex(i,j,k, q,r+1,0, im, ac_lmax_prod)] += 
                  r/2./eta*(vrr_terms[iindex(i,j,k, q,r-1,0, im, ac_lmax_prod)]
                	        - Z_ZE * vrr_terms[iindex(i,j,k, q,r-1,0, im+1, ac_lmax_prod)]);
			  }

              if (j>0)
			  {
                vrr_terms[iindex(i,j,k, q,r+1,0, im, ac_lmax_prod)] += 
                  j/2.* inv_ZE *vrr_terms[iindex(i,j-1,k,q,r,0,im+1, ac_lmax_prod)];
			  }
            }
          }
        }
      }
    }
  }

  for (s=0; s<nc; ++s)
  {
    for (r=0; r<mc+1; ++r)
	{
      for (q=0; q<lc+1; ++q)
	  {
        for (k=0; k<na+1; ++k)
		{
          for (j=0; j<ma+1; ++j)
		  {
            for (i=0; i<la+1; ++i)
			{
              for (im=0; im<mtot-i-j-k-q-r-s; ++im)
			  {
                vrr_terms[iindex(i,j,k, q,r,s+1, im, ac_lmax_prod)] = 
                  (qz-zc)*vrr_terms[iindex(i,j,k, q,r,s, im, ac_lmax_prod)]
                  + (wz-qz)*vrr_terms[iindex(i,j,k, q,r,s, im+1, ac_lmax_prod)];

                if (s>0)
				{
                  vrr_terms[iindex(i,j,k, q,r,s+1, im, ac_lmax_prod)] += 
                    s/2./eta*(vrr_terms[iindex(i,j,k, q,r,s-1, im, ac_lmax_prod)]
                	          - Z_ZE * vrr_terms[iindex(i,j,k, q,r,s-1, im+1, ac_lmax_prod)]);
				}

                if (k>0)
				{
                  vrr_terms[iindex(i,j,k, q,r,s+1, im, ac_lmax_prod)] += 
                    k/2.* inv_ZE *vrr_terms[iindex(i,j,k-1, q,r,s, im+1, ac_lmax_prod)];
				}
              }
            }
          }
        }
      }
    }
  }

  val = vrr_terms[iindex(la,ma,na, lc,mc,nc, m, ac_lmax_prod)];

  free(ac_lmax_prod);
  free(vrr_terms);

  return val;
}

// Convert the 7-dimensional indices to a 1d iindex 
int iindex(int la, int ma, int na, int lc, int mc, int nc, int m, int *ac_lmax_prod)
{
  return la + ma * ac_lmax_prod[0] + na * ac_lmax_prod[1] + 
         lc * ac_lmax_prod[2] + mc * ac_lmax_prod[3] + nc * ac_lmax_prod[4] + 
         m * ac_lmax_prod[5];
}
