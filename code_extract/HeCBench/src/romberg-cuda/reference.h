//
// https://en.wikipedia.org/wiki/Romberg%27s_method
//

#ifdef DEBUG
void dump_row(size_t i, double *R) {
  printf("R[%2zu] = ", i);
  for (size_t j = 0; j <= i; ++j){
    printf("%f ", R[j]);
  }
  printf("\n");
}
#endif

/* 
 f: function to integrate
 a: lower limit
 b: upper limit
 acc: desired accuracy
*/
double reference (double (*f)(double), double a, double b, size_t max_steps, double acc) {
  double R1[max_steps], R2[max_steps]; // buffers
  double *Rp = &R1[0], *Rc = &R2[0]; // Rp is previous row, Rc is current row
  double h = (b-a); //step size
  Rp[0] = (f(a) + f(b))*h*.5; // first trapezoidal step

#ifdef DEBUG
  dump_row(0, Rp);
#endif

  for (size_t i = 1; i < max_steps; ++i) {
    h /= 2.;
    double c = 0;
    size_t ep = 1 << (i-1); //2^(n-1)
    for (size_t j = 1; j <= ep; ++j) {
      c += f(a+(2*j-1)*h);
    }
    Rc[0] = h*c + .5*Rp[0]; //R(i,0)

    for (size_t j = 1; j <= i; ++j) {
      double n_k = pow(4, j);
      Rc[j] = (n_k*Rc[j-1] - Rp[j-1])/(n_k-1); // compute R(i,j)
    }

    // Dump ith row of R, R[i,i] is the best estimate so far
#ifdef DEBUG
    dump_row(i, Rc);
#endif

    if (i > 1 && fabs(Rp[i-1]-Rc[i]) < acc) {
      return Rc[i-1];
    }

    // swap Rn and Rc as we only need the last row
    double *rt = Rp;
    Rp = Rc;
    Rc = rt;
  }
  return Rp[max_steps-1]; // return our best guess
}


