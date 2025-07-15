__device__
float LCG_random_float(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

__device__
void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

__global__
void setupKernel(unsigned int* state) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  state[idx] = idx;
  for (int i = 0; i < idx; i++)
    LCG_random_init(&state[idx]);
}

__device__
void decrypt(const int* encrypted, const int* key, int* decrypted) {

  int columns[KEY_LENGTH][SECTION_CONSTANT+1];
  int offset = 0;
  int colLength[KEY_LENGTH];

  for (int j=0; j<KEY_LENGTH; ++j) {
    colLength[j] = ENCRYPTEDLEN / KEY_LENGTH;
    if (j < ENCRYPTEDLEN % KEY_LENGTH)
      colLength[j]++;
  }

  for (int keyPos=0; keyPos < KEY_LENGTH; ++keyPos) {
    offset = 0;
    for (int i=0; i<KEY_LENGTH; ++i)
      if (key[i] < key[keyPos])
        offset += colLength[i];

    for (int j=0; j<colLength[keyPos]; ++j)   
      columns[key[keyPos]][j] = encrypted[offset+j];          
  } 

  for (int j=0; j<ENCRYPTEDLEN; ++j) 
    decrypted[j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];  
} 

__device__
void swapElements(int *key, int posLeft, int posRight) {
  if (posLeft != posRight)
  {
    key[posLeft] -= key[posRight];
    key[posRight] += key[posLeft];
    key[posLeft] = key[posRight] - key[posLeft];
  }
}

__device__ 
void swapBlock(int *key, int posLeft, int posRight, int length) {  
  for (int i=0; i<length; i++) 
    swapElements(key, (posLeft+i)%KEY_LENGTH, (posRight+i)%KEY_LENGTH);
}

__global__ 
void decode(const float *__restrict d_scores, 
            const int *__restrict d_encrypted,
            const unsigned int*__restrict  globalState, 
            int *__restrict d_decrypted) {

  __shared__ float shared_scores[ALPHABET*ALPHABET];

  int key[KEY_LENGTH];
  int localDecrypted[ENCRYPTEDLEN];  
  int bestLocalDecrypted[ENCRYPTEDLEN];  
  int leftLetter = 0;
  int rightLetter = 0;
  int backupKey[KEY_LENGTH];
  int shiftHelper[KEY_LENGTH];
  int blockStart, blockEnd;
  int l,f,t,t0,n,ff,tt;
  float tempScore = 0.f;
  float bestScore = CAP;
  int j = 0, jj = 0;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int localState = globalState[idx];

  if (threadIdx.x == 0) {
    for (j=0; j<ALPHABET;++j)
      for (jj=0; jj<ALPHABET; ++jj)
        shared_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
  }

  __syncthreads();

  for (j=0; j<KEY_LENGTH; ++j) 
    key[j]=j;

  for (j=0; j<KEY_LENGTH; ++j) {
    swapElements(key, j, LCG_random_float(&localState)*KEY_LENGTH);
  }

  for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

    for (j=0; j<KEY_LENGTH;j++)
      backupKey[j] = key[j];

    tempScore = 0.f;

    int branch = LCG_random_float(&localState)*100; 

    if (branch < HEUR_THRESHOLD_OP1)
    {
      for (j=0; j<1+LCG_random_float(&localState)*OP1_HOP; j++) 
      {
        leftLetter = LCG_random_float(&localState)*KEY_LENGTH;   
        rightLetter = LCG_random_float(&localState)*KEY_LENGTH; 
        swapElements(key, leftLetter, rightLetter);
      }            
    }

    else if (branch < HEUR_THRESHOLD_OP2)
    {
      for (j=0; j< 1+LCG_random_float(&localState)*OP2_HOP;j++)
      {
        blockStart = LCG_random_float(&localState)*KEY_LENGTH;
        blockEnd = LCG_random_float(&localState)*KEY_LENGTH;
        swapBlock(key, blockStart, blockEnd, 1+LCG_random_float(&localState)*(abs((blockStart-blockEnd))-1));
      }
    }

    else 
    {
      l = 1 + LCG_random_float(&localState)*(KEY_LENGTH-2);
      f = LCG_random_float(&localState)*(KEY_LENGTH-1);
      t = (f+1+(LCG_random_float(&localState)*(KEY_LENGTH-2)));
      t = t % KEY_LENGTH;

      for (j=0; j< KEY_LENGTH;j++)
        shiftHelper[j] = key[j];

      t0 = (t-f+KEY_LENGTH) % KEY_LENGTH;
      n = (t0+l) % KEY_LENGTH;

      for (j=0; j<n;j++) 
      {
        ff = (f+j) % KEY_LENGTH;
        tt = (((t0+j)%n)+f)%KEY_LENGTH;
        key[tt] = shiftHelper[ff];
      }        
    }      

    decrypt(d_encrypted, key, localDecrypted);    

    for (j=0; j<ENCRYPTEDLEN-1; ++j) {
      tempScore += shared_scores[ALPHABET*localDecrypted[j] + localDecrypted[j+1]];
    }

    if (tempScore < bestScore) {
      bestScore = tempScore;
      for (j=0; j<ENCRYPTEDLEN; ++j) {
        bestLocalDecrypted[j] = localDecrypted[j];
      }
    }    

    else 
    {
      for (j=0; j<KEY_LENGTH;j++)
        key[j] = backupKey[j];      
    }
  }

  for (j=0; j<ENCRYPTEDLEN; ++j)
    d_decrypted[idx*ENCRYPTEDLEN+j] = bestLocalDecrypted[j];
}

