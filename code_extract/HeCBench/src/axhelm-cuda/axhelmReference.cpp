#define cubNq3 (cubNq*cubNq*cubNq)
#define Nq3 (Nq*Nq*Nq)

static int meshIJN(const int i, const int j, const int N){

  return i + j*N;

}

static int meshIJKN(const int i, const int j, const int k, const int N){

  return i + j*N + k*N*N;

}

/* comment the unused function
static int meshIJKLN(const int i, const int j, const int k, const int l, const int N){

  return i + j*N + k*N*N + l*N*N*N;

}
*/

static void axhelmElementReference(int cubNq,
     		            int element,
		            dfloat lambda,
		            const dfloat *  ggeo,
		            const dfloat *  cubD,
		            const dfloat * qIII,
		            dfloat *lapqIII){
  
  dfloat Gqr[cubNq][cubNq][cubNq];
  dfloat Gqs[cubNq][cubNq][cubNq];
  dfloat Gqt[cubNq][cubNq][cubNq];

  for(int k=0;k<cubNq;++k){
    for(int j=0;j<cubNq;++j){
      for(int i=0;i<cubNq;++i){
	
	dfloat qr = 0;
	dfloat qs = 0;
	dfloat qt = 0;
	
	for(int n=0;n<cubNq;++n){
	  int in = meshIJN(n,i,cubNq);
	  int jn = meshIJN(n,j,cubNq);
	  int kn = meshIJN(n,k,cubNq);
	  
	  int kjn = meshIJKN(n,j,k,cubNq);
	  int kni = meshIJKN(i,n,k,cubNq);
	  int nji = meshIJKN(i,j,n,cubNq);
	  
	  qr += cubD[in]*qIII[kjn];
	  qs += cubD[jn]*qIII[kni];
	  qt += cubD[kn]*qIII[nji];	  
	}

	const int gbase = element*p_Nggeo*cubNq3 + meshIJKN(i,j,k,cubNq);
	
	dfloat G00 = ggeo[gbase+p_G00ID*cubNq3];
	dfloat G01 = ggeo[gbase+p_G01ID*cubNq3];
	dfloat G02 = ggeo[gbase+p_G02ID*cubNq3];
	dfloat G11 = ggeo[gbase+p_G11ID*cubNq3];
	dfloat G12 = ggeo[gbase+p_G12ID*cubNq3];
	dfloat G22 = ggeo[gbase+p_G22ID*cubNq3];
	
	Gqr[k][j][i] = (G00*qr + G01*qs + G02*qt);
	Gqs[k][j][i] = (G01*qr + G11*qs + G12*qt);
	Gqt[k][j][i] = (G02*qr + G12*qs + G22*qt);
      }
    }
  }


  for(int k=0;k<cubNq;++k){
    for(int j=0;j<cubNq;++j){
      for(int i=0;i<cubNq;++i){
  
	int kji = meshIJKN(i,j,k,cubNq);
	
	const int gbase = element*p_Nggeo*cubNq3 + meshIJKN(i,j,k,cubNq);

	dfloat GWJ = ggeo[gbase+p_GWJID*cubNq3];
	dfloat lapq = lambda*GWJ*qIII[kji];
	
	for(int n=0;n<cubNq;++n){
	  int ni = meshIJN(i,n,cubNq);
	  int nj = meshIJN(j,n,cubNq);
	  int nk = meshIJN(k,n,cubNq);

	  lapq += cubD[ni]*Gqr[k][j][n];
	  lapq += cubD[nj]*Gqs[k][n][i];
	  lapq += cubD[nk]*Gqt[n][j][i];	  
	}
	
	lapqIII[kji] = lapq;
      }
    }
  }
}

void axhelmReference(int Nq,
		     const int numElements,
		     dfloat lambda,
		     const dfloat *  ggeo,
		     const dfloat *  D,
		     const dfloat *  solIn,
		     dfloat *  solOut){

  for(int e=0;e<numElements;++e)
    axhelmElementReference(Nq, e, lambda, ggeo, D, solIn+e*Nq3, solOut+e*Nq3);
  
}


