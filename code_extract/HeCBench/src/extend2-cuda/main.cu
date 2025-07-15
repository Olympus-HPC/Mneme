/* The MIT License
   Copyright (c) 2011 by Attractive Chaos <attractor@live.co.uk>
   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:
   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "read_data.h"

static void check(int a, int b, const char *s)
{
  if (a != b) printf("Error: %s %d %d\n", s, a, b);
}

typedef struct {
  int h, e;
} eh_t;

__global__ void
kernel_extend2(
    const unsigned char* query,
    const unsigned char* target,
    const char* mat,
    eh_t* eh,
    char* qp,
    int* qle_acc,
    int* tle_acc,
    int* gtle_acc,
    int* gscore_acc,
    int* max_off_acc,
    int* score_acc,
    const int qlen, 
    const int tlen, 
    const int m, 
    const int o_del, 
    const int e_del, 
    const int o_ins, 
    const int e_ins, 
    int w, 
    const int end_bonus, 
    const int zdrop, 
    const int h0)
{
  int oe_del = o_del + e_del;
  int oe_ins = o_ins + e_ins; 
  int i, j, k;
  int beg, end;
  int max, max_i, max_j, max_ins, max_del, max_ie;
  int gscore;
  int max_off;

  // generate the query profile
  for (k = i = 0; k < m; ++k) {
    const char *p = mat + k * m;
    for (j = 0; j < qlen; ++j)
      qp[i++] = p[query[j]];
  }

  // fill the first row
  eh[0].h = h0; 
  eh[1].h = h0 > oe_ins? h0 - oe_ins : 0;

  for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
    eh[j].h = eh[j-1].h - e_ins;

  // adjust $w if it is too large
  k = m * m;
  for (i = 0, max = 0; i < k; ++i) // get the max score
    max = max > mat[i]? max : mat[i];
  max_ins = (int)((float)(qlen * max + end_bonus - o_ins) / e_ins + 1.f);
  max_ins = max_ins > 1? max_ins : 1;
  w = w < max_ins? w : max_ins;
  max_del = (int)((float)(qlen * max + end_bonus - o_del) / e_del + 1.f);
  max_del = max_del > 1? max_del : 1;
  w = w < max_del? w : max_del; // TODO: is this necessary?
  // DP loop
  max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
  max_off = 0;
  beg = 0, end = qlen;
  for (i = 0; i < tlen; ++i) {
    int t, f = 0, h1, m = 0, mj = -1;
    char *q = qp + target[i] * qlen;

    // apply the band and the constraint (if provided)
    if (beg < i - w) beg = i - w;
    if (end > i + w + 1) end = i + w + 1;
    if (end > qlen) end = qlen;

    // compute the first column
    if (beg == 0) {
      h1 = h0 - (o_del + e_del * (i + 1));
      if (h1 < 0) h1 = 0;
    } 
    else 
      h1 = 0;

    for (j = beg; j < end; ++j) {
      // At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
      // Similar to SSE2-SW, cells are computed in the following order:
      //   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
      //   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
      //   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
      eh_t *p = eh+j;
      int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
      p->h = h1;          // set H(i,j-1) for the next row
      M = M? M + q[j] : 0;// separating H and M to disallow a cigar like "100M3I3D20M"
      h = M > e? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
      h = h > f? h : f;
      h1 = h;             // save H(i,j) to h1 for the next column
      mj = m > h? mj : j; // record the position where max score is achieved
      m = m > h? m : h;   // m is stored at eh[mj+1]
      t = M - oe_del;
      t = t > 0? t : 0;
      e -= e_del;
      e = e > t? e : t;   // computed E(i+1,j)
      p->e = e;           // save E(i+1,j) for the next row
      t = M - oe_ins;
      t = t > 0? t : 0;
      f -= e_ins;
      f = f > t? f : t;   // computed F(i,j+1)
    }
    eh[end].h = h1; eh[end].e = 0;
    if (j == qlen) {
      max_ie = gscore > h1? max_ie : i;
      gscore = gscore > h1? gscore : h1;
    }
    if (m == 0) break;
    if (m > max) {
      max = m, max_i = i, max_j = mj;
      max_off = max_off > abs(mj - i)? max_off : abs(mj - i);
    } else if (zdrop > 0) {
      if (i - max_i > mj - max_j) {
        if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
      } else {
        if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
      }
    }
    // update beg and end for the next round
    for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j);
    beg = j;
    for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j);
    end = j + 2 < qlen? j + 2 : qlen;
    //beg = 0; end = qlen; // uncomment this line for debugging
  }
  *qle_acc = max_j + 1;
  *tle_acc = max_i + 1;
  *gtle_acc = max_ie + 1;
  *gscore_acc = gscore;
  *max_off_acc = max_off;
  *score_acc = max;
}

float extend2(struct extend2_dat *d)
{
  eh_t *eh = NULL; /* score array*/
  char *qp = NULL; /* query profile*/
  posix_memalign((void**)&eh, 64, (d->qlen+1) * 8);
  posix_memalign((void**)&qp, 64, d->qlen * d->m);
  memset(eh, 0, (d->qlen+1) * 8);

  int qle, tle, gtle, gscore, max_off, score;

  const int qlen = d->qlen;
  const int tlen = d->tlen;
  const int m = d->m;
  const int o_del = d->o_del; 
  const int e_del = d->e_del; 
  const int o_ins = d->o_ins; 
  const int e_ins = d->e_ins; 
  const int w = d->w;
  const int end_bonus = d->end_bonus;
  const int zdrop = d->zdrop;
  const int h0 = d->h0;

  auto start = std::chrono::steady_clock::now();

  unsigned char *d_query;
  cudaMalloc((void**)&d_query, qlen);
  cudaMemcpyAsync(d_query, d->query, qlen, cudaMemcpyHostToDevice, 0);

  unsigned char *d_target;
  cudaMalloc((void**)&d_target, tlen);
  cudaMemcpyAsync(d_target, d->target, tlen, cudaMemcpyHostToDevice, 0);

  char *d_mat;
  cudaMalloc((void**)&d_mat, m*m);
  cudaMemcpyAsync(d_mat, d->mat, m*m, cudaMemcpyHostToDevice, 0);

  eh_t *d_eh;
  cudaMalloc((void**)&d_eh, (qlen+1)*sizeof(eh_t));
  cudaMemcpyAsync(d_eh, eh, (qlen+1)*sizeof(eh_t), cudaMemcpyHostToDevice, 0);

  char *d_qp;
  cudaMalloc((void**)&d_qp, qlen*m);
  cudaMemcpyAsync(d_qp, qp, qlen*m, cudaMemcpyHostToDevice, 0);

  int *d_qle;
  cudaMalloc((void**)&d_qle, 4);

  int *d_tle;
  cudaMalloc((void**)&d_tle, 4);

  int *d_gtle;
  cudaMalloc((void**)&d_gtle, 4);

  int *d_gscore;
  cudaMalloc((void**)&d_gscore, 4);

  int *d_max_off;
  cudaMalloc((void**)&d_max_off, 4);

  int *d_score;
  cudaMalloc((void**)&d_score, 4);

  kernel_extend2<<<1,1>>>(
      d_query,
      d_target,
      d_mat,
      d_eh,
      d_qp,
      d_qle,
      d_tle,
      d_gtle,
      d_gscore,
      d_max_off,
      d_score,
      qlen,
      tlen,
      m, 
      o_del, 
      e_del, 
      o_ins, 
      e_ins, 
      w, 
      end_bonus, 
      zdrop, 
      h0);

  cudaMemcpyAsync(&qle, d_qle, 4, cudaMemcpyDeviceToHost, 0);
  cudaMemcpyAsync(&tle, d_tle, 4, cudaMemcpyDeviceToHost, 0);
  cudaMemcpyAsync(&gtle, d_gtle, 4, cudaMemcpyDeviceToHost, 0);
  cudaMemcpyAsync(&max_off, d_max_off, 4, cudaMemcpyDeviceToHost, 0);
  cudaMemcpyAsync(&gscore, d_gscore, 4, cudaMemcpyDeviceToHost, 0);
  cudaMemcpyAsync(&score, d_score, 4, cudaMemcpyDeviceToHost, 0);

  cudaFree(d_query);
  cudaFree(d_target);
  cudaFree(d_mat);
  cudaFree(d_eh);
  cudaFree(d_qp);
  cudaFree(d_qle);
  cudaFree(d_tle);
  cudaFree(d_gtle);
  cudaFree(d_gscore);
  cudaFree(d_max_off);
  cudaFree(d_score);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  check(d->qle, qle, "qle");
  check(d->tle, tle, "tle");
  check(d->gtle, gtle, "gtle");
  check(d->gscore, gscore, "gscore");
  check(d->max_off, max_off, "max_off");
  check(d->score, score, "score");

  free(eh);
  free(qp);

#ifdef VERBOSE
  printf("device: qle=%d, tle=%d, gtle=%d, gscore=%d, max_off=%d, score=%d\n",
      qle, tle, gtle, gscore, max_off, score);
#endif

 return time;
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  struct extend2_dat d;

  // Instead of iterating over a directory, list the file names (17 in total)
  const char* files[] = {
#include "filelist.txt"
  };

  float time = 0.f;
  for (int f = 0; f < repeat; f++) {
    read_data(files[f%17], &d);
    time += extend2(&d);
  }
  printf("Average offload time %f (us)\n", (time * 1e-3f) / repeat);
  return 0;
}
