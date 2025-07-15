#ifndef __MESH_BASIS
#define __MESH_BASIS

// extern "C" is required
extern "C"
{
void dgesv_ (int *NrowsA, int *NcolsA, double *A, int *LDA, int *ipiv,  double *B, int *LDB, int *info);
void dgeev_ (char *JOBVL, char *JOBVR, int *N, double *A, int *LDA, double *WR, double *WI, double *VL, int *LDVL, double *VR, int *LDVR, double *WORK, int *LWORK, int *INFO );
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}


void matrixInverse(int N, dfloat *A);
void matrixEig(int N, dfloat *A, dfloat *VR, dfloat *WR, dfloat *WI);

void readDfloatArray(FILE *fp, const char *label, dfloat **A, int *Nrows, int* Ncols);
void readIntArray(FILE *fp, const char *label, int **A, int *Nrows, int* Ncols);

void meshMassMatrix(int Np, dfloat *V, dfloat **MM);
int meshVandermonde1D(int N, int Npoints, dfloat *r, dfloat **V, dfloat **Vr);

dfloat mygamma(dfloat x);
int meshJacobiGQ(dfloat alpha, dfloat beta, int N, dfloat **x, dfloat **w);
int meshJacobiGL(dfloat alpha, dfloat beta, int N, dfloat **x, dfloat **w);

dfloat meshJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);
dfloat meshGradJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

void meshOrthonormalBasis1D(dfloat a, int i, dfloat *P, dfloat *Pr);
void meshOrthonormalBasisTri2D(dfloat a, dfloat b, int i, int j, dfloat *P, dfloat *Pr, dfloat *Ps);
void meshOrthonormalBasisQuad2D(dfloat a, dfloat b, int i, int j, dfloat *P, dfloat *Pr, dfloat *Ps);
void meshOrthonormalBasisTet3D(dfloat a, dfloat b, dfloat c, int i, int j, int k, dfloat *P, dfloat *Pr, dfloat *Ps, dfloat *Pt);
void meshOrthonormalBasisHex3D(dfloat a, dfloat b, dfloat c, int i, int j, int k, dfloat *P, dfloat *Pr, dfloat *Ps, dfloat *Pt);

int meshVandermonde1D(int N, int Npoints, dfloat *r, dfloat **V, dfloat **Vr);
int meshVandermondeTri2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat **V, dfloat **Vr, dfloat **Vs);
int meshVandermondeQuad2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat **V, dfloat **Vr, dfloat **Vs);
int meshVandermondeTet3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t,
    dfloat **V, dfloat **Vr, dfloat **Vs, dfloat **Vt);
int meshVandermondeHex3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t, dfloat **V, dfloat **Vr, dfloat **Vs, dfloat **Vt);

void matrixRightSolve(int NrowsA, int NcolsA, dfloat *A, int NrowsB, int NcolsB, dfloat *B, dfloat *C);
void matrixEig(int N, dfloat *A, dfloat *VR, dfloat *WR, dfloat *WI);
void matrixPrint(FILE *fp, const char *mess, int Nrows, int Ncols, dfloat *A);
void matrixCompare(FILE *fp, const char *mess, int Nrows, int Ncols, dfloat *A, dfloat *B);

void meshDmatrix1D(int N, int Npoints, dfloat *r, dfloat **Dr);
void meshDmatricesTri2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat **Dr, dfloat **Ds);
void meshDmatricesQuad2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat **Dr, dfloat **Ds);
void meshDmatricesTet3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t, dfloat **Dr, dfloat **Ds, dfloat **Dt);
void meshDmatricesHex3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t, dfloat **Dr, dfloat **Ds, dfloat **Dt);

void meshContinuousFilterMatrix1D(int N, int Nlow, dfloat *r, dfloat **F);

void meshInterpolationMatrix1D(int N,
    int NpointsIn, dfloat *rIn, 
    int NpointsOut, dfloat *rOut,
    dfloat **I);

void meshInterpolationMatrixTri2D(int N,
    int NpointsIn, dfloat *rIn, dfloat *sIn,
    int NpointsOut, dfloat *rOut, dfloat *sOut,
    dfloat **I);

void meshInterpolationMatrixTet3D(int N,
    int NpointsIn,  dfloat *rIn,  dfloat *sIn,  dfloat *tIn,
    int NpointsOut, dfloat *rOut, dfloat *sOut, dfloat *tOut,
    dfloat **I);

void meshMassMatrix(int Np, dfloat *V, dfloat **MM);

void meshLiftMatrixTri2D(int N, int Np, int *faceNodes, dfloat *r, dfloat *s, dfloat **LIFT);
void meshLiftMatrixTet3D(int N, int Np, int *faceNodes, dfloat *r, dfloat *s, dfloat *t, dfloat **LIFT);

void meshCubatureWeakDmatrices1D(int N, int Np, dfloat *V,
    int cubNp, dfloat *cubr, dfloat *cubw,
    dfloat **cubDrT, dfloat **cubProject);

void meshCubatureWeakDmatricesTri2D(int N, int Np, dfloat *V,
    int cubNp, dfloat *cubr, dfloat *cubs, dfloat *cubw,
    dfloat **cubDrT, dfloat **cubDsT, dfloat **cubProject);

void meshCubatureWeakDmatricesTet3D(int N, int Np, dfloat *V,
    int cubNp, dfloat *cubr, dfloat *cubs, dfloat *cubt, dfloat *cubw,
    dfloat **cubDrT, dfloat **cubDsT, dfloat **cubDtT, dfloat **cubProject);

int meshCubatureSurfaceMatricesTri2D(int N, int Np, dfloat *r, dfloat *s, dfloat *V, int *faceNodes,
    int intNfp,  dfloat **intInterp, dfloat **intLIFT);


void meshReferenceBK1(int Nq, int cubNq, const int numElements, const dfloat *ggeo, const dfloat *INToC, const dfloat *q,  dfloat * __restrict__ Aq);

void meshReferenceBK3(int Nq,
    int cubNq,
    const int numElements,
    dfloat lambda,
    const dfloat *  ggeo,
    const dfloat *  INToC,
    const dfloat *  cubD,
    const dfloat *  solIn,
    dfloat *  solOut);

void meshReferenceBK5(int Nq,
    int Nelements,
    dfloat lambda,
    const dfloat *  ggeo,
    const dfloat *  cubD,
    const dfloat * qIII,
    dfloat *lapqIII);
#endif
