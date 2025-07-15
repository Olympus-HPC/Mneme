__global__ void
kernel(float* node, 
    float* R6, 
    const int Imin,
    const int Jmin,
    const int Imax,
    const int Jmax,
    const int NLat,
    const float Par_sshZeroThreshold,
    const float Par_sshArrivalThreshold,
    const int Par_time)
{
  int j = blockDim.x*blockIdx.x+threadIdx.x + Jmin;
  int i = blockDim.y*blockIdx.y+threadIdx.y + Imin;
  if (i <= Imax && j <= Jmax) {
    int m = idx(j,i);
    if( Node(m, iD) == 0 ) return;
    Node(m, iH) = Node(m, iH) - Node(m, iR1)*( Node(m, iM) - Node(m-NLat, iM) + Node(m, iN)*R6[j] - Node(m-1, iN)*R6[j-1] );

    float absH = fabs(Node(m, iH));

    if( absH < Par_sshZeroThreshold ) Node(m, iH) = 0.;

    if( Node(m, iH) > Node(m, iHmax) ) Node(m, iHmax) = Node(m, iH);

    if( Par_sshArrivalThreshold && Node(m, iTime) < 0 && absH > Par_sshArrivalThreshold ) 
      Node(m, iTime) = (float)Par_time;
  }
}

 __global__ void
kernel2(float* node, 
    const float* C1,
    const float* C2,
    const float* C3,
    const float* C4,
    const int Imin,
    const int Jmin,
    const int Imax,
    const int Jmax,
    const int NLat,
    const int NLon )
{
  int i, j, m;
  // open bondary conditions
  if( Jmin <= 2 ) {
    for( i=2; i<=(NLon-1); i++ ) {
      m = idx(1,i);
      Node(m, iH) = sqrt( pow(Node(m, iN), 2.0f) + 
          0.25f*pow((Node(m, iM)+Node(m-NLat, iM)),2.0f) )*C1[i];
      if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Imin <= 2 ) {
    for( j=2; j<=(NLat-1); j++ ) {
      m = idx(j,1);
      Node(m, iH) = sqrt( pow(Node(m, iM),2.0f) + 
          0.25f*pow((Node(m, iN)+Node(m-1, iN)),2.0f) )*C2[j];
      if( Node(m, iM) > 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Jmax >= (NLat-1) ) {
    for( i=2; i<=(NLon-1); i++ ) {
      m = idx(NLat,i);
      Node(m, iH) = sqrt( pow(Node(m-1, iN),2.0f) + 0.25f*pow((Node(m, iM)+Node(m-1, iM)),2.0f) )*C3[i];
      if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Imax >= (NLon-1) ) {
    for( j=2; j<=(NLat-1); j++ ) {
      m = idx(j,NLon);
      Node(m, iH) = sqrt( pow(Node(m-NLat, iM),2.0f) + 0.25f*pow((Node(m, iN)+Node(m-1, iN)),2.0f) )*C4[j];
      if( Node(m-NLat, iM) < 0 ) Node(m, iH) = - Node(m, iH);
    }
  }
  if( Jmin <= 2 ) {
    m = idx(1,1);
    Node(m, iH) = sqrt( pow(Node(m, iM),2.0f) + pow(Node(m, iN),2.0f) )*C1[1];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(1,NLon);
    Node(m, iH) = sqrt( pow(Node(m-NLat, iM),2.0f) + pow(Node(m, iN),2.0f) )*C1[NLon];
    if( Node(m, iN) > 0 ) Node(m, iH) = - Node(m, iH);
  }
  if( Jmin >= (NLat-1) ) {
    m = idx(NLat,1);
    Node(m, iH) = sqrt( pow(Node(m, iM),2.0f) + pow(Node(m-1, iN),2.0f) )*C3[1];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
    m = idx(NLat,NLon);
    Node(m, iH) = sqrt( pow(Node(m-NLat, iM),2.0f) + pow(Node(m-1, iN),2.0f) )*C3[NLon];
    if( Node(m-1, iN) < 0 ) Node(m, iH) = - Node(m, iH);
  }
}

__global__ void
kernel3(float* node, 
    float* R6, 
    const int Imin,
    const int Jmin,
    const int Imax,
    const int Jmax,
    const int NLat)
{
  int j = blockDim.x*blockIdx.x+threadIdx.x + Jmin;
  int i = blockDim.y*blockIdx.y+threadIdx.y + Imin;
  if (i <= Imax && j <= Jmax) {
    int m = idx(j,i);
    if( (Node(m, iD)*Node(m+NLat, iD)) != 0 )
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH)-Node(m, iH));

    if( (Node(m, iD)*Node(m+1, iD)) != 0 )
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH)-Node(m, iH));
  }
}

__global__ void
kernel4(float* node, 
    const float* C1,
    const float* C2,
    const float* C3,
    const float* C4,
    int *Imin,
    int *Jmin,
    int *Imax,
    int *Jmax,
    const int NLat,
    const int NLon,
    const float Par_sshClipThreshold )
{
  int i, j, m;
  int enlarge;
  if( Jmin[0] <= 2 ) {
    for( i=1; i<=(NLon-1); i++ ) {
      m = idx(1,i);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }
  if( Imin[0] <= 2 ) {
    for( j=1; j<=NLat; j++ ) {
      m = idx(j,1);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }
  if( Jmax[0] >= (NLat-1) ) {
    for( i=1; i<=(NLon-1); i++ ) {
      m = idx(NLat,i);
      Node(m, iM) = Node(m, iM) - Node(m, iR2)*(Node(m+NLat, iH) - Node(m, iH));
    }
  }
  if( Imin[0] <= 2 ) {
    for( j=1; j<=(NLat-1); j++ ) {
      m = idx(j,1);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }
  if( Jmin[0] <= 2 ) {
    for( i=1; i<=NLon; i++ ) {
      m = idx(1,i);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }
  if( Imax[0] >= (NLon-1) ) {
    for( j=1; j<=(NLat-1); j++ ) {
      m = idx(j,NLon);
      Node(m, iN) = Node(m, iN) - Node(m, iR4)*(Node(m+1, iH) - Node(m, iH));
    }
  }

  // calculation area for the next step
  if( Imin[0] > 2 ) {
    for( enlarge=0, j=Jmin[0]; j<=Jmax[0]; j++ ) {
      if( fabs(Node(idx(j,Imin[0]+2), iH)) > Par_sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imin[0]--; if( Imin[0] < 2 ) Imin[0] = 2; }
  }
  if( Imax[0] < (NLon-1) ) {
    for( enlarge=0, j=Jmin[0]; j<=Jmax[0]; j++ ) {
      if( fabs(Node(idx(j,Imax[0]-2), iH)) > Par_sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Imax[0]++; if( Imax[0] > (NLon-1) ) Imax[0] = NLon-1; }
  }
  if( Jmin[0] > 2 ) {
    for( enlarge=0, i=Imin[0]; i<=Imax[0]; i++ ) {
      if( fabs(Node(idx(Jmin[0]+2,i), iH)) > Par_sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmin[0]--; if( Jmin[0] < 2 ) Jmin[0] = 2; }
  }
  if( Jmax[0] < (NLat-1) ) {
    for( enlarge=0, i=Imin[0]; i<=Imax[0]; i++ ) {
      if( fabs(Node(idx(Jmax[0]-2,i), iH)) > Par_sshClipThreshold ) { enlarge = 1; break; }
    }
    if( enlarge ) { Jmax[0]++; if( Jmax[0] > (NLat-1) ) Jmax[0] = NLat-1; }
  }
}
