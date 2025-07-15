inline __device__ uchar4 operator^(uchar4 a, uchar4 b)
{
  return make_uchar4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

inline __device__ void operator^=(uchar4 &a, const uchar4 b)
{
  a.x ^= b.x;
  a.y ^= b.y;
  a.z ^= b.z;
  a.w ^= b.w;
}

__host__ __device__
uchar galoisMultiplication(uchar a, uchar b)
{
  uchar p = 0; 
  for(unsigned int i=0; i < 8; ++i)
  {
    if((b&1) == 1)
    {
      p^=a;
    }
    uchar hiBitSet = (a & 0x80);
    a <<= 1;
    if(hiBitSet == 0x80)
    {
      a ^= 0x1b;
    }
    b >>= 1;
  }
  return p;
}

inline __device__
uchar4 sboxRead(const uchar * SBox, uchar4 block)
{
  return make_uchar4(SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]);
}

__device__
uchar4 mixColumns(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
  unsigned int bw = 4;

  uchar x, y, z, w;

  x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
  y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
  z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
  w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);

  for(unsigned int k=1; k< 4; ++k)
  {
    x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
    y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
    z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
    w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
  }

  return make_uchar4(x, y, z, w);
}

__device__
uchar4 shiftRows(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    //r.xyzw() = r.yzwx();
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = make_uchar4(y,z,w,x);
  }
  return r;
}

__global__
void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                const uint     width , 
                const uint     rounds )

{
  __shared__ uchar4 block0[4];
  __shared__ uchar4 block1[4];

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  //unsigned int localIdx = threadIdx.x;
  unsigned int localIdy = threadIdx.y;

  unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
  unsigned int localIndex  = localIdy;

  uchar4 galiosCoeff[4];
  galiosCoeff[0] = make_uchar4(2, 0, 0, 0);
  galiosCoeff[1] = make_uchar4(3, 0, 0, 0);
  galiosCoeff[2] = make_uchar4(1, 0, 0, 0);
  galiosCoeff[3] = make_uchar4(1, 0, 0, 0);

  block0[localIndex]  = input[globalIndex];

  block0[localIndex] ^= roundKey[localIndex];

  for(unsigned int r=1; r < rounds; ++r)
  {
    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

    __syncthreads();
    block1[localIndex]  = mixColumns(block0, galiosCoeff, localIndex); 

    __syncthreads();
    block0[localIndex] = block1[localIndex]^roundKey[r*4 + localIndex];
  }
  block0[localIndex] = sboxRead(SBox, block0[localIndex]);

  block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

  output[globalIndex] =  block0[localIndex]^roundKey[(rounds)*4 + localIndex];
}

__device__
uchar4 shiftRowsInv(uchar4 row, unsigned int j)
{
  uchar4 r = row;
  for(uint i=0; i < j; ++i)  
  {
    // r = r.wxyz();
    uchar x = r.x;
    uchar y = r.y;
    uchar z = r.z;
    uchar w = r.w;
    r = make_uchar4(w,x,y,z);
  }
  return r;
}

__global__
void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                const  uint    width , 
                const  uint    rounds)

{
  __shared__ uchar4 block0[4];
  __shared__ uchar4 block1[4];

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  //unsigned int localIdx = threadIdx.x;
  unsigned int localIdy = threadIdx.y;

  unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
  unsigned int localIndex  = localIdy;

  uchar4 galiosCoeff[4];
  galiosCoeff[0] = make_uchar4(14, 0, 0, 0);
  galiosCoeff[1] = make_uchar4(11, 0, 0, 0);
  galiosCoeff[2] = make_uchar4(13, 0, 0, 0);
  galiosCoeff[3] = make_uchar4( 9, 0, 0, 0);

  block0[localIndex]  = input[globalIndex];

  block0[localIndex] ^= roundKey[4*rounds + localIndex];

  for(unsigned int r=rounds -1 ; r > 0; --r)
  {
    block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    __syncthreads();
    block1[localIndex] = block0[localIndex]^roundKey[r*4 + localIndex];

    __syncthreads();
    block0[localIndex]  = mixColumns(block1, galiosCoeff, localIndex); 
  }  

  block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

  block0[localIndex] = sboxRead(SBox, block0[localIndex]);

  output[globalIndex] =  block0[localIndex]^roundKey[localIndex];
}
