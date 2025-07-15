void cpu_kernel(
  const unsigned int rows, 
  const unsigned int cols, 
  const int cRows, 
  const int contRows,
  const unsigned char *snpdata,
  float *results)
{
  unsigned char y;
  int m, n;
  unsigned int p = 0;
  int cases[3];
  int controls[3];
  int tot_cases = 1;
  int tot_controls= 1;
  int total = 1;
  float chisquare = 0.0f;
  float exp[3];
  float Conexpected[3];
  float Cexpected[3];
  float numerator1;
  float numerator2;

  for(unsigned tid = 0; tid < cols; tid++) {

    chisquare=0;
    cases[0]=1;cases[1]=1;cases[2]=1;
    controls[0]=1;controls[1]=1;controls[2]=1;

    for ( m = 0 ; m < cRows ; m++ ) {
      y = snpdata[(size_t) m * (size_t) cols + tid];
      if ( y == '0') { cases[0]++; }
      else if ( y == '1') { cases[1]++; }
      else if ( y == '2') { cases[2]++; }
    }

    for ( n = cRows ; n < cRows + contRows ; n++ ) {
      y = snpdata[(size_t) n * (size_t) cols + tid];
      if ( y == '0' ) { controls[0]++; }
      else if ( y == '1') { controls[1]++; }
      else if ( y == '2') { controls[2]++; }
    }

    tot_cases = cases[0]+cases[1]+cases[2];
    tot_controls = controls[0]+controls[1]+controls[2];
    total = tot_cases + tot_controls;

    for( p = 0 ; p < 3; p++) {
      exp[p] = (float)cases[p] + controls[p];
      Cexpected[p] = tot_cases * exp[p] / total;
      Conexpected[p] = tot_controls * exp[p] / total;
      numerator1 = (float)cases[p] - Cexpected[p];
      numerator2 = (float)controls[p] - Conexpected[p];
      chisquare += numerator1 * numerator1 / Cexpected[p] +  numerator2 * numerator2 / Conexpected[p];
    }
    results[tid] = chisquare;
  }
}
