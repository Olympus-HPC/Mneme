void reference(const float *__restrict subject,
               const float *__restrict avgs,
               const float *__restrict stds, 
                     float *__restrict lb_keogh,
               const float *__restrict zlower,
               const float *__restrict zupper,
               const int M,
               const int N) 
{
  // calculate LB_Keogh
  for (int indx = 0; indx < N-M+1; indx++) {

    // obtain statistics
    float residues= 0;
    float avg = avgs[indx];
    float std = stds[indx];

    for (int i = 0; i < M; ++i) {

      // differences to envelopes
      float value = (subject[indx+i]-avg)/std;
      float lower = value-zlower[i];
      float upper = value-zupper[i];

      // Euclidean or Manhattan distance?
      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb_keogh[indx] = residues;
  }
}

