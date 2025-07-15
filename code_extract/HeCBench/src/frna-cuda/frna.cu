#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "cu.h"
#include "frna.h"
#include "util.h"

/* penalty for a helix terminated by a pair containing a U */
DEV static int_t terminal_U_penalty(const fbase_t *s, const int i, const int j, fparam_t p)
{
  return s[i] == U || s[j] == U ? p->terminal_AU_penalty : 0;
}

DEV static int_t dangle_3p_energy(const fbase_t *s,
    const int i,
    const int j,
    const int ip1,
    fparam_t p)
{
  return p->dangle_3p[s[i]][s[j]][s[ip1]] + terminal_U_penalty(s,i,j,p);
}

DEV static int_t dangle_5p_energy(const fbase_t *s,
    const int i,
    const int j,
    const int jm1,
    fparam_t p)
{
  return p->dangle_5p[s[i]][s[j]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}

DEV static int_t terminal_stack(const fbase_t *s,
    const int i,
    const int j,
    const int ip1,
    const int jm1,
    fparam_t p)
{
  return p->tstack[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}

DEV static int_t terminal_stack_multibranch(const fbase_t *s,
    const int i,
    const int j,
    const int ip1,
    const int jm1,
    fparam_t p)
{
  return p->tstackm[s[i]][s[j]][s[ip1]][s[jm1]] + terminal_U_penalty(s,i,j,p);
}

DEV static const int_t *lookup_find(const fbase_t *s, const int d, fparam_t p)
{
  int i;
  switch (d) {
    case 3:
      for (i = 0; i < p->ntriloop; i++)
        if (sequences_match(s, p->triloop[i].seq, d+2))
          return &p->triloop[i].val;
      break;
    case 4:
      for (i = 0; i < p->ntloop; i++)
        if (sequences_match(s, p->tloop[i].seq, d+2))
          return &p->tloop[i].val;
      break;
    case 6:
      for (i = 0; i < p->nhexaloop; i++)
        if (sequences_match(s, p->hexaloop[i].seq, d+2))
          return &p->hexaloop[i].val;
      break;
  }
  return 0;
}

/***
 * Energy of a hairpin loop with d unpaired bases, d = j-i-1
 * s[i] is paired with s[j]
 * s[i+1] is mismatched with s[j-1]
 ***/
DEV static int_t hairpin_loop_energy(const fbase_t *s,
    const int i,
    const int j,
    const int d,
    fparam_t p)
{
  /* Lookup tables for special hairpin loops */
  const int_t *val;
  if ((val = lookup_find(&s[i],d,p)))
    return *val;

  /* Hairpin loop initiation penalty */
  int_t e;
  if (d > LOOP_MAX)
    e = (int_t) (p->hairpin_loop_initiation[LOOP_MAX] + p->prelog *
        LOG((float) d / LOOP_MAX));
  else
    e = p->hairpin_loop_initiation[d];

  if (d == 3) {
    if (contains_only_base(C,d,&s[i+1]))
      e += p->c_hairpin_of_3;
    e += terminal_U_penalty(s,i,j,p);
  } else {
    e += p->tstackh[s[i]][s[j]][s[i+1]][s[j-1]];
    if (contains_only_base(C,d,&s[i+1]))
      e += p->c_hairpin_slope*d + p->c_hairpin_intercept;
  }

  if (s[i] == G && s[j] == U && i > 1 && s[i-1] == G && s[i-2] == G)
    e += p->bonus_for_GGG_hairpin;

  return e;

}

DEV static int_t real_min(int_t a, int_t b) { return a < b ? a : b; }

/***
 * Energy of an internal/bulge loop with d1, d2 unpaired bases,
 *   d1 = ip-i-1,  d2 = j-jp-1
 * s[i] is paired with s[j]
 * s[i+1] is mismatched with s[j-1]
 * s[ip-1] is mismatched with s[jp+1]
 * s[ip] is paired with s[jp]
 ***/

DEV static int_t alternative_bulge_loop_correction (const int n, const fbase_t *s,
    const int i,
    const int ip) //i<ip
{
  int count = 1;
  int k;
  //float result;
  if (i!=n-1){
    k = i;
    while (k>=0 && s[k]==s[i+1]) {
      count++;
      k--;
    }

    k = ip;
    while (k<=n-1 && (s[k]==s[i+1])) {
      count++;
      k++;
    }
  }
  return (int_t) (-1.0f * RT * conversion_factor * log ((float) count));
}

DEV static int_t internal_loop_energy(const fbase_t *s,
    const int n,
    const int i,
    const int j,
    const int ip,
    const int jp,
    const int d1,
    const int d2,
    fparam_t p)
{
  /* Bulge loops */
  if (d1 == 0 || d2 == 0) {
    int_t e = p->bulge_loop_initiation[d1+d2];
    if (d1 == 1 || d2 == 1) { /* single-nucleotide bulge */
      e += p->stack[s[i]][s[j]][s[ip]][s[jp]];
      if (d1==0) e += alternative_bulge_loop_correction(n,s,jp,j); //correction for multiple equivalent bulge loops
      //else e += alternative_bulge_loop_correction(s,i,jp);
      else e += alternative_bulge_loop_correction(n,s,i,ip);
      if ((d1 == 1 && s[i+1] == C && (s[i] == C || s[i+2] == C)) ||
          (d2 == 1 && s[j-1] == C && (s[j] == C || s[j-2] == C)))
        e += p->Bonus_for_Single_C_bulges_adjacent_to_C;
    } else {
      e += terminal_U_penalty(s,i,j,p);
      e += terminal_U_penalty(s,ip,jp,p);
    }
    return e;
  }

  /* Small internal loops */
  if (d1 == 1 && d2 == 1)
    return p->int11[s[i]][s[i+1]][s[i+2]][s[j-2]][s[j-1]][s[j]];
  if (d1 == 2 && d2 == 2)
    return p->int22[s[i]][s[ip]][s[j]][s[jp]][s[i+1]][s[i+2]][s[j-1]][s[j-2]];
  if (d1 == 1 && d2 == 2)
    return p->int21[s[i]][s[j]][s[i+1]][s[j-1]][s[jp+1]][s[ip]][s[jp]];
  if (d1 == 2 && d2 == 1)
    return p->int21[s[jp]][s[ip]][s[jp+1]][s[ip-1]][s[i+1]][s[j]][s[i]];

  /* Larger internal loops */
  tab4_t *sp;
  if (d1 == 1 || d2 == 1)
    sp = &p->tstacki1n;
  else if ((d1 == 2 && d2 == 3) || (d1 == 3 && d2 == 2))
    sp = &p->tstacki23;
  else
    sp = &p->tstacki;
  return p->internal_loop_initiation[d1+d2] +
    real_min(p->fm_array_first_element*abs(d1-d2), p->maximum_correction) +
    (*sp)[s[i]][s[j]][s[i+1]][s[j-1]] +
    (*sp)[s[jp]][s[ip]][s[jp+1]][s[ip-1]];

}

DEV static int_t coaxial_flush(const fbase_t *s,
    const int i,
    const int j,
    const int ip,
    const int jp,
    fparam_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->coaxial[s[i]][s[j]][s[ip]][s[jp]];
}

DEV static int_t coaxial_mismatch1(const fbase_t *s,
    const int i,
    const int j,
    const int ip,
    const int jp,
    fparam_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->tstackcoax[s[j]][s[i]][s[j+1]][s[i-1]] +
    p->coaxstack[s[j+1]][s[i-1]][s[ip]][s[jp]];
}

DEV static int_t coaxial_mismatch2(const fbase_t *s,
    const int i,
    const int j,
    const int ip,
    const int jp,
    fparam_t p)
{
  return terminal_U_penalty(s,i,j,p) + terminal_U_penalty(s,ip,jp,p) +
    p->tstackcoax[s[jp]][s[ip]][s[jp+1]][s[ip-1]] +
    p->coaxstack[s[j]][s[i]][s[j+1]][s[jp+1]];
}

DEV static void free_energy_min(int_t *a, const int_t b)
{
  if(*a>b) *a = b;
}

DEV static int int_min(int a, int b) { return a < b ? a : b; }

DEV static int_t int_t_min(int_t a, int_t b) { return a < b ? a : b; }

DEV HOST static int ind(int i, int j, int n)
{
  return i*n + j;
}

DEV HOST inline static int cp(int i, int j, const fbase_t *s)
{
  return j-i-1 >= LOOP_MIN && is_canonical_pair(s[i],s[j]);
}

DEV HOST inline static int can_pair(int i, int j, int n, const fbase_t *s)
{
  if (j < i) {
    const int tmp = i;
    i = j;
    j = tmp;
  }
  return cp(i,j,s) && ((i > 0 && j < n-1 && cp(i-1,j+1,s)) || cp(i+1,j-1,s));
}

DEV HOST inline static int not_isolated(int i,int j,int n, const fbase_t *s)
{
  if (j < i) {
    const int tmp = i;
    i = j;
    j = tmp;
  }
  return is_canonical_pair(s[i],s[j]) && ((i > 0 && j < n-1 && cp(i-1,j+1,s)) || cp(i+1,j-1,s));
}

DEV static int wrap(int i, int n)
{
  return i >= n ? i-n : i;
}

DEV static int is_exterior(int i, int j)
{
  return j < i;
}

DEV static int is_interior(int i, int j)
{
  return i < j;
}

DEV static int_t *array_val(int_t *a, int i, int j, int n, const fbase_t *s)
{
  return can_pair(i,j,n,s) ? &a[ind(i,j,n)] : 0;
}

#ifdef __CUDACC__
#define ISTART blockIdx.x
#define IINC gridDim.x
#else
#define ISTART 0
#define IINC 1
#endif

//MFE recursions begin
//TODO
//figure out source of differences in arrays
//integrate with rnastructure traceback


//when recursions work on the cpu:
//do the same thing with the calculation on the GPU
GLOBAL static void calc_V_hairpin_and_V_stack
(int d, 
 int n, 
 const fbase_t *__restrict s,
 int_t *__restrict v,
 const fparam_t __restrict p)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if((is_interior(i,j) && !can_pair(i,j,n,s)) || (is_exterior(i,j) && (!is_canonical_pair(s[i],s[j]) ))){
      v[ind(i,j,n)] = INF; //this is important
      continue;
    }
    int_t vij = INF; //temp variable to fold free energy sum
    if (i != n-1 && j != 0) {
      /* hairpin loop */
      if (is_interior(i,j))
        vij = hairpin_loop_energy(s,i,j,d,p);
      /* stack */
      if (can_pair(i+1,j-1,n,s) && !((is_interior(i,j)) && (d <= LOOP_MIN-2)))//-2???
        free_energy_min(&vij, p->stack[s[i]][s[j]][s[i+1]][s[j-1]] + v[ind(i+1,j-1,n)]);
    }
    v[ind(i,j,n)] = vij;
  }
}

#ifdef __CUDACC__

#define NTHREAD 256
#define SQRT_NTHREAD 16

DEV static void free_energy_min_reduce(int_t *x, int tid, int nt)
{
  __shared__ int_t buf[NTHREAD];
  buf[tid] = *x;
  for (nt /= 2, __syncthreads(); nt > 0; nt /= 2, __syncthreads())
    if (tid < nt)
      free_energy_min(&buf[tid], buf[tid+nt]);
  if (tid == 0)
    *x = buf[0];
}

#endif /* __CUDACC__ */
GLOBAL static void calc_V_bulge_internal (
  int d, 
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  const fparam_t __restrict__ p)
{
  //  Vbi(i,j) = min[V(k,l)+ Ebulge/int(i,j,k,l)] where i<k<l<j, i!=i+1, and j!=j-1
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) ||
        (is_interior(i,j) && d <= LOOP_MIN+2) ||
        !can_pair(i,j,n,s))
      continue;
    int_t vij = INF;
#ifdef __CUDACC__
    const int d1start = threadIdx.x;
    const int d1inc = blockDim.x;
#else
    const int d1start = 0;
    const int d1inc = 1;
#endif
    const int dmax = int_min(LOOP_MAX, d-2);
    const int d1max = int_min(dmax, n-i-2);
    int d1;
    for (d1 = d1start; d1 <= d1max; d1 += d1inc) { //d1start is threadid, d1max is max loop size
      const int ip = i+d1+1; //ip depends on thread's ID in x dimension
      const int d2max = int_min(dmax-d1, j-1);
#ifdef __CUDACC__
      const int d2start = d1 > 0 ? threadIdx.y : threadIdx.y + 1;
      const int d2inc = blockDim.y;
#else
      const int d2start = d1 > 0 ? 0 : 1;
      const int d2inc = 1;
#endif
      int d2;
      for (d2 = d2start; d2 <= d2max; d2 += d2inc) {
        const int jp = j-d2-1;//jp depends on thread's ID in the y dimension
        if (can_pair(ip,jp,n,s))
          free_energy_min(&vij, internal_loop_energy(s,n,i,j,ip,jp,d1,d2,p) + v[ind(ip,jp,n)]);
      }
    }
#ifdef __CUDACC__
    const int tid = threadIdx.x * blockDim.y + threadIdx.y;
    free_energy_min_reduce(&vij, tid, blockDim.x*blockDim.y); //after we have 1 value per thread, do parallel reduction
    if (tid != 0)
      continue;
#endif
    free_energy_min(&v[ind(i,j,n)], vij); //write vij to V
  }
}

GLOBAL static void calc_V_multibranch (
  int d,
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  const int_t *__restrict__ wm,
  const fparam_t __restrict__ p)
{

  //  Vmb(i,j) = min[WM(i+1,j-1)+c+a, WM(i+2,j-1)+Edangle5'+a+b+c, WM(i+1,j-2)+Edangle3'+a+b+c, WM(i+2,j-2)+Edangleboth+a+2b+c,
  //    min_over_k[ V(i+1,k) + min[W(k+1,j-1), WM(k+1,j-1)]] + a+2c+Eflushcoax(i to j, i+1 to k) , //various coaxial stacking possibilities
  //    min_over_k[ V(k,j-1) + min[W(i+1,k-1), WM(i+1,k-1)]] + a+2c+Eflushcoax(i to j, k to j-1) ,
  //    min_over_k[ V(i+2,k) + min[W(k+2,j-1), WM(k+2,j-1)]] + a+2c+2b+Emismatch3'coax(i to j, i+2 to k) ,
  //    min_over_k[ V(i+2,k) + min[W(k+1,j-2), WM(k+1,j-2)]] + a+2c+2b+Emismatch5'coax(i to j, i+2 to k) ,
  //    min_over_k[ V(k,j-2) + min[W(i+2,k-1), WM(i+2,k-1)]] + a+2c+2b+Emismatch3'coax(i to j, k to j-2) ,
  //    min_over_k[ V(k,j-2) + min[W(i+1,k-2), WM(i+1,k-2)]] + a+2c+2b+Emismatch5'coax(i to j, k to j-2) ]

  //  where i < k < j


  //V(i,j) = min(V(i,j), Vmb(i,j))

  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) || !can_pair(i,j,n,s))
      continue;
    int_t vij=INF;
    if (d > 2*LOOP_MIN + 3 && i != n-1 && j != 0) { //if i and j are far enough apart to close a MBL..
      free_energy_min(&vij, wm[ind(i+1,j-1,n)] + terminal_U_penalty(s,i,j,p) + p->a + p->c);
      if (i != n-2)
        free_energy_min(&vij, wm[ind(i+2,j-1,n)] + dangle_3p_energy(s,i,j,i+1,p) + p->a + p->b + p->c);
      if (j != 1)
        free_energy_min(&vij, wm[ind(i+1,j-2,n)] + dangle_5p_energy(s,i,j,j-1,p) + p->a + p->b + p->c);
      if (i != n-2 && j != 1)
        free_energy_min(&vij, wm[ind(i+2,j-2,n)] + terminal_stack_multibranch(s,i,j,i+1,j-1,p) + p->a + 2*p->b + p->c);
    }
    free_energy_min(&v[ind(i,j,n)], vij);
  }
}

GLOBAL static void calc_V_exterior (
  int d, 
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  const int_t *__restrict__ w5,
  const int_t *__restrict__ w3,
  const fparam_t __restrict__ p)
{
  //  Vexterior(i,j) = min[ W3(i+1)+W3(j-1-N), W3(i+2)+W5(j-1-N)+E5'dangle, W3(i+1)+W5(j-2-N)+E3'dangle, W3(i+2)+W5(j-2-N)+Emismatch,
  //    min_over_k[ V(i+1,k) + W3(k+1) + W5(j-1-N) + Eflushcoax ],
  //    min_over_k[ V(k,j-1-N) + W3(i+1) + W5(k-1) + E ],
  //    min_over_k[ V(i+2,k-2) + W3(k+1) + W5(j-1-N) + E ],
  //    min_over_k[ V(i+2,k-1) + W3(k+1) + W5(j-2-N) + E ],
  //    min_over_k[ V(k+1,j-2-N) + W3(i+1) + W5(k-1) + E ],
  //    min_over_k[ V(k,j-2-N) + W3(i+2) + W5(k-1) + E ] ]

  int i;
  for (i = ISTART; i < n; i += IINC) { 
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ( is_interior(i,j))
      continue;
    int_t vij = INF; //temp variable to fold free energy sum
    if(is_canonical_pair(s[i],s[j])&&not_isolated(i,j,n,s)){
      free_energy_min(&vij, w3[i+1] + w5[j-1] + terminal_U_penalty(s,i,j,p));
      if (i != n-1)
        free_energy_min(&vij, w3[i+2] + w5[j-1] + dangle_3p_energy(s,i,j,i+1,p));
      if (j != 0)
        free_energy_min(&vij, w3[i+1] + w5[j-2] + dangle_5p_energy(s,i,j,j-1,p));
      if (i != n-1 && j != 0)
        free_energy_min(&vij, w3[i+2] + w5[j-2] + terminal_stack(s,i,j,i+1,j-1,p));
    }
    free_energy_min(&v[ind(i,j,n)], vij);
  }
}

GLOBAL static void calc_W (
  int d,
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  int_t *__restrict__ w,
  const fparam_t __restrict__ p)
{
  //W(i,j) = min[V(i,j)+c,V(i+1,j)+Edangle5',
  //      V(i,j+1)+Edangle3',
  //      V(i+1,j+1)+Edangleboth]

  int i;
  for (i = ISTART; i < n; i += IINC) { 
    const int jtmp = i+d+1;  // max: n-1+n-2+1  
    const int j = wrap(jtmp,n); // n-2
    int_t wij = INF; //temp variable to fold free energy sum
    int_t* v_temp;
    //consider adding nucleotide to existing loop
    if(d>0){
      if (i!=n-1)
        free_energy_min(&wij, w[ind(i+1,j,n)] + p->b);
      if(j!=0)
        free_energy_min(&wij, w[ind(i,j-1,n)] + p->b);
    }
    if((is_interior(i,j) && (d>LOOP_MIN-1))){
      v_temp = array_val(v,i,j,n,s);
      free_energy_min(&wij, (v_temp? *v_temp:INF) + terminal_U_penalty(s,i,j,p) + p->c);
      if(j!=0){
        v_temp = array_val(v,i,j-1,n,s);
        free_energy_min(&wij, (v_temp? *v_temp:INF) + dangle_3p_energy(s,j-1,i,j,p) + p->b + p->c);
      }

      if(i!=n-1) {
        v_temp = array_val(v,i+1,j,n,s);
        free_energy_min(&wij, (v_temp? *v_temp:INF) + dangle_5p_energy(s,j,i+1,i,p) + p->b + p->c);
      }

      if((i!=n-1) && (j!=0)){
        v_temp = array_val(v,i+1,j-1,n,s);
        free_energy_min(&wij, (v_temp? *v_temp:INF) + terminal_stack_multibranch(s,j-1,i+1,j,i,p) + 2*p->b + p->c);
      }
    }
    if(is_exterior(i,j)){
      free_energy_min(&wij, v[ind(i,j,n)] + terminal_U_penalty(s,i,j,p) + p->c);
      if(j!=0){
        free_energy_min(&wij, v[ind(i,j-1,n)] + dangle_3p_energy(s,j-1,i,j,p) + p->b + p->c);
      }

      if(i!=n-1) {
        free_energy_min(&wij, v[ind(i+1,j,n)] + dangle_5p_energy(s,j,i+1,i,p) + p->b + p->c);
      }

      if((i!=n-1) && (j!=0)){
        free_energy_min(&wij, v[ind(i+1,j-1,n)] + terminal_stack_multibranch(s,j-1,i+1,j,i,p) + 2*p->b + p->c);
      }
    }
    w[ind(i,j,n)] = wij;
  }
}

GLOBAL static void calc_WM (
  int d,
  int n,
  const fbase_t *__restrict__ s,
  int_t *__restrict__ w,
  int_t *__restrict__ wm,
  const fparam_t __restrict__ p)
{
  //WM(i,j) = min[W(i,k)+W(k+1,j),
  //      V(i,k)+V(k+1,j)+2c+Eflushcoax,
  //      V(i,k)+V(k+2,j-1)+2c+Ecoax5'mismatch,
  //      V(i+1,k)+V(k+2,j)+2c+Ecoax3'mismatch]

  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    int_t tmp = INF;

    //don't need to calculate every WM
    if((is_interior(i,j) && (j-i-1 <= 2*LOOP_MIN+2))){//condition copied verbatim from algorithm.cpp
      wm[ind(i,j,n)]=INF;
      continue;
    }

#ifdef __CUDACC__
    const int kstart = i + threadIdx.x;
    const int kinc = blockDim.x;
#else
    const int kstart = i;
    const int kinc = 1;
#endif
    int ktmp;
    for (ktmp = kstart; ktmp < jtmp; ktmp += kinc) {
      if (ktmp != n-1) {
        const int k = wrap(ktmp,n);
        free_energy_min(&tmp, w[ind(i,k,n)] + w[ind(k+1,j,n)]);
      }
    }

    if(d>0){
      if (i!=n-1)
        free_energy_min(&tmp, wm[ind(i+1,j,n)] + p->b);
      if(j!=0)
        free_energy_min(&tmp, wm[ind(i,j-1,n)] + p->b);
    }

#ifdef __CUDACC__
    free_energy_min_reduce(&tmp, threadIdx.x, blockDim.x);
    if (threadIdx.x != 0)
      continue;
#endif
    wm[ind(i,j,n)] = tmp;
    free_energy_min(&w[ind(i,j,n)],tmp);
  }
}

GLOBAL static void calc_coaxial (
  int d, 
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  const int_t *__restrict__ w,
  const int_t *__restrict__ w5,
  const int_t *__restrict__ w3,
  const fparam_t __restrict__ p)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) || !can_pair(i,j,n,s))
      continue;
    const int_t *v1;
    int_t vij = INF;
    /* exterior */
    if (is_exterior(i,j)) {
      int k, kstart;
#ifdef __CUDACC__
      kstart = threadIdx.x;
      const int kinc = blockDim.x;
#else
      kstart = 0;
      const int kinc = 1;
#endif
      for (k = kstart; k < j - LOOP_MIN; k += kinc) {
        if ((v1 = array_val(v,k,j-1,n,s)))
          free_energy_min(&vij, w3[i+1] + w5[k-1] + coaxial_flush(s,k,j-1,j,i,p) + (*v1));
        if (j-2 >= 0) {
          if (i < n-1 && (v1 = array_val(v,k,j-2,n,s)))
            free_energy_min(&vij, w3[i+2] + w5[k-1] + coaxial_mismatch2(s,k,j-2,j,i,p) + (*v1));
          if ((v1 = array_val(v,k+1,j-2,n,s)))
            free_energy_min(&vij, w3[i+1] + w5[k-1] + coaxial_mismatch1(s,k+1,j-2,j,i,p) + (*v1));
        }
      }
#ifdef __CUDACC__
      kstart = i+LOOP_MIN+1 + threadIdx.x;
#else
      kstart = i+LOOP_MIN+1;
#endif
      for (k = kstart; k < n; k += kinc) {
        if ((v1 = array_val(v,i+1,k,n,s)))
          free_energy_min(&vij, w3[k+1] + w5[j-1] + coaxial_flush(s,j,i,i+1,k,p) + (*v1));
        if (j > 0 && (v1 = array_val(v,i+2,k,n,s)))
          free_energy_min(&vij, w3[k+1] + w5[j-2] + coaxial_mismatch1(s,j,i,i+2,k,p) + (*v1));
        if ((v1 = array_val(v,i+2,k-1,n,s)))
          free_energy_min(&vij, w3[k+1] + w5[j-1] + coaxial_mismatch2(s,j,i,i+2,k-1,p) + (*v1));
      }
    } /* end exterior */

    /* multibranch */
    if (d > 2*LOOP_MIN + 3 && i != n-1 && j != 0) {
      int ktmp;
#ifdef __CUDACC__
      int ktmpstart = i+2 + threadIdx.x;
      const int ktmpinc = blockDim.x;
#else
      int ktmpstart = i+2;
      const int ktmpinc = 1;
#endif
      for (ktmp = ktmpstart; ktmp < jtmp-2; ktmp += ktmpinc) {
        const int k = wrap(ktmp,n);
        if (k != n-1) {
          if ((v1 = array_val(v,i+1,k,n,s)))
            free_energy_min(&vij, coaxial_flush(s,j,i,i+1,k,p) + (*v1) + p->a_2c +
                w[ind(k+1,j-1,n)]);
          if (ktmp+2 < jtmp-1 && i+1 != n-1 && k+1 != n-1 && (v1 = array_val(v,i+2,k,n,s))) {
            const int_t tmp = (*v1) + p->a_2b_2c;
            free_energy_min(&vij, coaxial_mismatch2(s,j,i,i+2,k,p) + tmp + w[ind(k+2,j-1,n)]);
            if (j != 1) {
              free_energy_min(&vij, coaxial_mismatch1(s,j,i,i+2,k,p) + tmp + w[ind(k+1,j-2,n)]);
            }
          }
        }
      }
#ifdef __CUDACC__
      ktmpstart = i+3 + threadIdx.x;
#else
      ktmpstart = i+3;
#endif
      for (ktmp = ktmpstart; ktmp < jtmp-1; ktmp += ktmpinc) {
        const int k = wrap(ktmp,n);
        if (k != 0) {
          if ((v1 = array_val(v,k,j-1,n,s)))
            free_energy_min(&vij, coaxial_flush(s,k,j-1,j,i,p) + (*v1) + p->a_2c +
                w[ind(i+1,k-1,n)]);
          if (j != 1 && ktmp > i+3 && (v1 = array_val(v,k,j-2,n,s))) {
            const int_t tmp = (*v1) + p->a_2b_2c;
            if (k != 1)
              free_energy_min(&vij, coaxial_mismatch1(s,k,j-2,j,i,p) + tmp + w[ind(i+1,k-2,n)]);
            if (i != n-2)
              free_energy_min(&vij, coaxial_mismatch2(s,k,j-2,j,i,p) + tmp + w[ind(i+2,k-1,n)]);
          }
        }
      }
    } /* end multibranch */
#ifdef __CUDACC__
    free_energy_min_reduce(&vij, threadIdx.x, blockDim.x);
    if (threadIdx.x != 0)
      continue;
#endif
    free_energy_min(&v[ind(i,j,n)], vij);
  } /* end loop over i */
} /* end calc_coaxial */

GLOBAL static void calc_wl_coax(
  int d, 
  int n, 
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  int_t *__restrict__ w,
  int_t *__restrict__ wm,
  const fparam_t __restrict__ p,
  int_t *__restrict__ wca)
{
  int i;
  for (i = ISTART; i < n; i += IINC) {
    const int jtmp = i+d+1;
    const int j = wrap(jtmp,n);
    if ((is_exterior(i,j) && i-j <= LOOP_MIN) ||
        (is_interior(i,j) && d <= 2*LOOP_MIN+1))
      continue;
#ifdef __CUDACC__
    const int kstart = i+LOOP_MIN+1 + threadIdx.x;
    const int kinc = blockDim.x;
#else
    const int kstart = i+LOOP_MIN+1;
    const int kinc = 1;
#endif
    int ktmp;
    int_t tmp1 = INF, tmp2 = INF;
    for (ktmp = kstart; ktmp < jtmp-LOOP_MIN-1; ktmp += kinc) {
      const int k = wrap(ktmp,n);
      if (k == n-1) continue;
      int_t *v1, *v2;
      if ((v1 = array_val(v,i,k,n,s)) && (v2 = array_val(v,k+1,j,n,s))){
        free_energy_min(&tmp1, (*v1) + (*v2) + coaxial_flush(s,i,k,k+1,j,p));
      }
      if (j == 0 || k+1 == n-1) continue;
      if (i != n-1 && (v1 = array_val(v,i+1,k,n,s)) && (v2 = array_val(v,k+2,j,n,s))){
        free_energy_min(&tmp2, (*v1) + (*v2) + coaxial_mismatch1(s,i+1,k,k+2,j,p));
      }
      if ((v1 = array_val(v,i,k,n,s)) && (v2 = array_val(v,k+2,j-1,n,s))){
        free_energy_min(&tmp2, (*v1) + (*v2) + coaxial_mismatch2(s,i,k,k+2,j-1,p));
      }
    }
#ifdef __CUDACC__
    free_energy_min_reduce(&tmp1, threadIdx.x, blockDim.x);
    free_energy_min_reduce(&tmp2, threadIdx.x, blockDim.x);
    if (threadIdx.x != 0) continue;
#endif
    wca[ind(i,j,n)] = int_t_min(tmp1,tmp2);
    free_energy_min(&wm[ind(i,j,n)], tmp1+2*p->c);
    free_energy_min(&wm[ind(i,j,n)], tmp2+2*p->b+2*p->c);
    free_energy_min(&w[ind(i,j,n)], wm[ind(i,j,n)]);
  } /* end loop over i */
} /* end calc_wl_coxial */


GLOBAL static void calc_w5_and_w3 (
  int d,
  int n,
  const fbase_t *__restrict__ s,
  int_t *__restrict__ v,
  int_t *__restrict__ w5,
  int_t *__restrict__ w3,
  const fparam_t __restrict__ p,
  const int_t *__restrict__ wca)
{
#ifdef __CUDACC__
    const int istart = threadIdx.x;
    const int iinc = blockDim.x;
#else
    const int istart = 0;
    const int iinc = 1;
#endif
    int_t w5tmp=0,w3tmp = 0;
    int i;
    int_t* v_temp;
    for (i = istart; i + LOOP_MIN <= d; i += iinc) {

      if((v_temp = array_val(v,i,d+1,n,s)))
        free_energy_min(&w5tmp, w5[i-1] + *v_temp + terminal_U_penalty(s,d+1,i,p)); //the nucleotide thats more 3' has to go first in terminal_U_penalty call
      if(d-i>LOOP_MIN){//necessary, or we seg fault because we try to have a pair in a 4mer
        if((v_temp = array_val(v,i,d,n,s)))
          free_energy_min(&w5tmp, w5[i-1] + *v_temp + dangle_3p_energy(s,d,i,d+1,p));
        if((v_temp = array_val(v,i+1,d+1,n,s)))
          free_energy_min(&w5tmp, w5[i-1] + *v_temp + dangle_5p_energy(s,d+1,i+1,i,p));
        free_energy_min(&w5tmp,w5[i-1] + wca[ind(i,d+1,n)]);
      }
      if ((d-i>LOOP_MIN+1) && ((v_temp = array_val(v,i+1,d,n,s))))
        free_energy_min(&w5tmp, w5[i-1] + *v_temp + terminal_stack(s,d,i+1,d+1,i,p));

      if((v_temp = array_val(v,n-d-2,n-i-1,n,s)))
        free_energy_min(&w3tmp, w3[n-i] + *v_temp + terminal_U_penalty(s,n-i-1,n-d-2,p));
      if((v_temp = array_val(v,n-d-2,n-i-2,n,s)))
        free_energy_min(&w3tmp, w3[n-i] + *v_temp + dangle_3p_energy(s,n-i-2,n-d-2,n-i-1,p));
      if((n-d-1 != 0) && ((v_temp = array_val(v,n-d-1,n-i-1,n,s))))
        free_energy_min(&w3tmp, w3[n-i] + *v_temp + dangle_5p_energy(s,n-i-1,n-d-1,n-d-2,p));
      if((n-i-2 != n-1) && (n-d-1 != 0) && ((v_temp = array_val(v,n-d-1,n-i-2,n,s))))
        free_energy_min(&w3tmp, w3[n-i] + *v_temp + terminal_stack(s,n-i-2,n-d-1,n-i-1,n-d-2,p));
      free_energy_min(&w3tmp,w3[n-i] + wca[ind(n-d-2,n-i-1,n)]);
    }
#ifdef __CUDACC__
    free_energy_min_reduce(&w5tmp, threadIdx.x, blockDim.x);
    free_energy_min_reduce(&w3tmp, threadIdx.x, blockDim.x);
    if (threadIdx.x != 0)
      return;
#endif
    w5[d+1] = w5[d];
    w3[n-d-2] = w3[n-d-1];
    free_energy_min(&w5[d+1], w5tmp);
    free_energy_min(&w3[n-d-2], w3tmp);
} /* end calc_w5_and_w3 */

GLOBAL static void init_w5_and_w3 (
  int n,
  int_t *__restrict__ w5,
  int_t *__restrict__ w3)
{
#ifdef __CUDACC__
  w5[blockIdx.x] = 0;
  w3[blockIdx.x] = 0;
#else
  int i;
  for(i=0;i<n+1;i++){
    w5[i] = 0;
    w3[i] = 0;
  }
#endif
}

//MFE recursions end

void initialize(int_t* arr,size_t size){
  size_t i;
  for(i=0;i<size;i++){
    arr[i] = INF;
  }
}

frna_t frna_new(const char *str, fparam_t par)
{
  frna_t p = (frna_t) safe_malloc(sizeof(struct frna));
  memset(p, 0, sizeof(struct frna));

  const int n = p->n = strlen(str);
  p->seq = fsequence_from_string(str);
  p->v = (int_t *) safe_malloc(n*n*sizeof(int_t));
  p->w = (int_t *) safe_malloc(n*n*sizeof(int_t));
  p->wm = (int_t *) safe_malloc(n*n*sizeof(int_t));
  p->wca = (int_t *) safe_malloc(n*n*sizeof(int_t));
  p->w5 = (int_t *) safe_malloc((n+1)*sizeof(int_t)) + 1;
  p->w3 = (int_t *) safe_malloc((n+1)*sizeof(int_t));
  initialize(p->v,n*n);
  initialize(p->w,n*n);
  initialize(p->wm,n*n);
  initialize(p->wca,n*n);

#ifdef __CUDACC__ /* do multithreaded fill on GPU */

#define ALLOC(a,sz) CU(cudaMalloc(&a,(sz)*sizeof(int_t)))

  int_t *v,*w,*wm,*w5,*w3,*wca;
  ALLOC(v,n*n); //best energy of structure closed by pair i,j. j>i: exterior fragment
  ALLOC(w,n*n); //best energy of structure from i to j
  ALLOC(wm,n*n); //best energy of structure i to j containing 2 or more branches
  ALLOC(w5,n+1); //best energy of structure from 1 to i
  w5++;//w5 is indexed from 1 -- is this a good idea?
  ALLOC(w3,n+1); //best energy of structure from i to numberofbases
  ALLOC(wca,n*n);

  fparam_t pm;
  CU(cudaMalloc(&pm, sizeof(struct fparam)));
  CU(cudaMemcpy(pm, par, sizeof(struct fparam), cudaMemcpyHostToDevice));

  fbase_t *s;
  CU(cudaMalloc(&s,n*sizeof(fbase_t)));
  CU(cudaMemcpy(s, p->seq, n*sizeof(fbase_t), cudaMemcpyHostToDevice));

  CU(cudaMemcpy(v, p->v, n*n*sizeof(int_t), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(w, p->w, n*n*sizeof(int_t), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(wm, p->wm, n*n*sizeof(int_t), cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  init_w5_and_w3<<<n,1>>>(n+1,w5-1,w3);

  for (int d = 0; d < n-1; d++) { //for fragment lengths (1 : n)
    calc_V_hairpin_and_V_stack<<<n,1>>>(d, n, s, v, pm);
    calc_V_bulge_internal<<<n,dim3(SQRT_NTHREAD,SQRT_NTHREAD,1)>>>(d, n, s, v, pm);
    calc_V_exterior<<<n,1>>>(d, n, s, v, w5, w3, pm);
    calc_V_multibranch<<<n,1>>>(d, n, s, v, wm, pm);
    calc_coaxial<<<n,NTHREAD>>>(d, n, s, v, w, w5, w3, pm);
    calc_W<<<n,1>>>(d, n, s, v, w, pm);
    calc_WM<<<n,NTHREAD>>>(d, n, s, w, wm, pm);
    calc_wl_coax<<<n,NTHREAD>>>(d, n, s, v, w, wm, pm, wca);
    calc_w5_and_w3<<<1,NTHREAD>>>(d, n, s, v, w5, w3, pm, wca);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time %f (s)\n", time * 1e-9f);

  CU(cudaMemcpy(p->v, v, n*n*sizeof(int_t), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(p->w, w, n*n*sizeof(int_t), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(p->wm, wm, n*n*sizeof(int_t), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(p->w5 - 1, w5 - 1, (n+1)*sizeof(int_t), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(p->w3, w3, (n+1)*sizeof(int_t), cudaMemcpyDeviceToHost));

  CU(cudaFree(v));
  CU(cudaFree(w5 - 1));
  CU(cudaFree(w3));
  CU(cudaFree(w));
  CU(cudaFree(wm));
  CU(cudaFree(pm));
  CU(cudaFree(s));

#else /* do serial fill on CPU */

#define ALLOC(a,sz) a = (int_t *) safe_malloc((sz)*sizeof(int_t))

  /*  ALLOC(v,n*n); //best energy of structure closed by pair i,j. j>i: exterior fragment
      ALLOC(w,n*n); //best energy of structure from i to j
      ALLOC(wm,n*n); //best energy of structure i to j containing 2 or more branches
      ALLOC(w5,n+1); //best energy of structure from 1 to i
      w5++;//w5 is indexed from 1 -- is this a good idea?
      ALLOC(w3,n+1); //best energy of structure from i to numberofbases
   */

  init_w5_and_w3(n,p->w5,p->w3);

  int d;

  for (d = 0; d < n-1; d++) {
    calc_V_hairpin_and_V_stack(d, p->n, p->seq, p->v, par); 
    calc_V_bulge_internal(d, p->n, p->seq, p->v, par);
    calc_V_exterior(d, p->n, p->seq, p->v, p->w5, p->w3, par);
    calc_V_multibranch(d, p->n, p->seq, p->v, p->wm, par);
    calc_coaxial(d, p->n, p->seq, p->v, p->w, p->w5, p->w3, par);
    calc_W(d, p->n, p->seq, p->v, p->w, par);
    calc_WM(d, p->n, p->seq, p->v, p->wm, par);
    calc_wl_coax(d, p->n, p->seq, p->v, p->w, p->wm, par, p->wca);
    calc_w5_and_w3(d, p->n, p->seq, p->v, p->w5, p->w3, par, p->wca);
  }
#endif /* __CUDACC__ */
  return p;
} /* end frna_new */

void frna_delete(frna_t p)
{
  if (p) {
    if (p->seq)
      free(p->seq);
    if (p->v)
      free(p->v);
    if (p->w)
      free(p->w);
    if (p->wm)
      free(p->wm);
    if (p->w5 - 1)
      free(p->w5 - 1);
    if (p->w3)
      free(p->w3);
    free(p);
  }
}

void frna_write(const frna_t p, const char* outfile )
{
  FILE *f = fopen(outfile,"w");
  if (!f) {
    printf("failed to open output file %s", outfile);
  }

  int i,j, n = p->n;
  const fbase_t *s = p->seq;
  fprintf(f, "n: %d\n", n);
  fprintf(f, "seq: ");
  for (i = 0; i < n; i++)
    fprintf(f, "%c", fbase_as_char(s[i]));
  fprintf(f, "\n");
  fprintf(f, "i\tj\tV:\tW:\tWM:\tV':\tW':\tWM':\n");
  for (j = 0; j < n; j++)
    for(i = 0; i < j; i++)
      fprintf(f, "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
       i+1,j+1,p->v[ind(i,j,n)],p->w[ind(i,j,n)],
       p->wm[ind(i,j,n)],p->v[ind(j,i,n)],p->w[ind(j,i,n)],p->wm[ind(j,i,n)] );

  fprintf(f, "\n\n\ni\tw5[i]\tw3[i]\n");
  fprintf(f, "0\t0\t0\n");
  for (i = 0; i < n; i++) {
    fprintf(f, "%d\t",i+1);
    fprintf(f, "%d\t",p->w5[i]);
    fprintf(f, "%d\n",p->w3[i]);
  }
}

short base_as_num(fbase_t b)
{
  switch (b) {
    case A:
      return 1;
    case C:
      return 2;
    case G:
      return 3;
    case U:
      return 4;
    default:
      printf("unknown base %d\n",b);
      die("base_as_num: unknown base");
      return 0;
  }
}

fbase_t num_as_base(short x)
{
  switch (x) {
    case 1:
      return A;
    case 2:
      return C;
    case 3:
      return G;
    case 4:
      return U;
    default:
      return A;
  }
}
