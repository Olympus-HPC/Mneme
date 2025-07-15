uchar getRconValue(unsigned int num)
{
  return Rcon[num];
}

uchar getSBoxValue(unsigned int num)
{
  return sbox[num];
}

uchar getSBoxInvert(unsigned int num)
{
  return rsbox[num];
}

void mixColumn(uchar *column)
{
  uchar cpy[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    cpy[i] = column[i];
  }
  column[0] = galoisMultiplication(cpy[0], 2)^
    galoisMultiplication(cpy[3], 1)^
    galoisMultiplication(cpy[2], 1)^
    galoisMultiplication(cpy[1], 3);

  column[1] = galoisMultiplication(cpy[1], 2)^
    galoisMultiplication(cpy[0], 1)^
    galoisMultiplication(cpy[3], 1)^
    galoisMultiplication(cpy[2], 3);

  column[2] = galoisMultiplication(cpy[2], 2)^
    galoisMultiplication(cpy[1], 1)^
    galoisMultiplication(cpy[0], 1)^
    galoisMultiplication(cpy[3], 3);

  column[3] = galoisMultiplication(cpy[3], 2)^
    galoisMultiplication(cpy[2], 1)^
    galoisMultiplication(cpy[1], 1)^
    galoisMultiplication(cpy[0], 3);
}

void mixColumnInv(uchar *column)
{
  uchar cpy[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    cpy[i] = column[i];
  }
  column[0] = galoisMultiplication(cpy[0], 14 )^
    galoisMultiplication(cpy[3], 9 )^
    galoisMultiplication(cpy[2], 13)^
    galoisMultiplication(cpy[1], 11);

  column[1] = galoisMultiplication(cpy[1], 14 )^
    galoisMultiplication(cpy[0], 9 )^
    galoisMultiplication(cpy[3], 13)^
    galoisMultiplication(cpy[2], 11);

  column[2] = galoisMultiplication(cpy[2], 14 )^
    galoisMultiplication(cpy[1], 9 )^
    galoisMultiplication(cpy[0], 13)^
    galoisMultiplication(cpy[3], 11);

  column[3] = galoisMultiplication(cpy[3], 14 )^
    galoisMultiplication(cpy[2], 9 )^
    galoisMultiplication(cpy[1], 13)^
    galoisMultiplication(cpy[0], 11);
}

void mixColumns(uchar * state, bool inverse)
{
  uchar column[4];
  for(unsigned int i=0; i < 4; ++i)
  {
    for(unsigned int j=0; j < 4; ++j)
    {
      column[j] = state[j*4 + i];
    }

    if(inverse)
    {
      mixColumnInv(column);
    }
    else
    {
      mixColumn(column);
    }

    for(unsigned int j=0; j < 4; ++j)
    {
      state[j*4 + i] = column[j];
    }
  }
}

void subBytes(uchar * state, bool inverse, unsigned int keySize)
{
  for(unsigned int i=0; i < keySize; ++i)
  {
    state[i] = inverse ? getSBoxInvert(state[i]): getSBoxValue(state[i]);
  }
}

void shiftRow(uchar *state, uchar nbr)
{
  for(unsigned int i=0; i < nbr; ++i)
  {
    uchar tmp = state[0];
    for(unsigned int j = 0; j < 3; ++j)
    {
      state[j] = state[j+1];
    }
    state[3] = tmp;
  }
}

void shiftRowInv(uchar *state, uchar nbr)
{
  for(unsigned int i=0; i < nbr; ++i)
  {
    uchar tmp = state[3];
    for(unsigned int j = 3; j > 0; --j)
    {
      state[j] = state[j-1];
    }
    state[0] = tmp;
  }
}

__host__
void shiftRows(uchar * state, bool inverse)
{
  for(unsigned int i=0; i < 4; ++i)
  {
    if(inverse)
      shiftRowInv(state + i*4, i);
    else
      shiftRow(state + i*4, i);
  }
}

void addRoundKey(uchar * state,
    uchar * rKey,
    unsigned int keySize)
{
  for(unsigned int i=0; i < keySize; ++i)
  {
    state[i] = state[i] ^ rKey[i];
  }
}


void aesRound(uchar * state, uchar * rKey, 
    bool decrypt, unsigned int keySize)
{
  subBytes(state, decrypt, keySize);
  shiftRows(state, decrypt);
  mixColumns(state, decrypt);
  addRoundKey(state, rKey, keySize);
}

void aesMain(uchar * state, uchar * rKey, unsigned int rounds, 
             bool decrypt, unsigned int keySize)
{
  addRoundKey(state, rKey, keySize);
  for(unsigned int i=1; i < rounds; ++i)
  {
    aesRound(state, rKey + keySize*i, decrypt, keySize);
  } 
  subBytes(state, decrypt, keySize);
  shiftRows(state, decrypt);
  addRoundKey(state, rKey + keySize*rounds, keySize);
}

void aesRoundInv(uchar * state, uchar * rKey, 
                 bool decrypt, unsigned int keySize)
{
  shiftRows(state, decrypt);
  subBytes(state, decrypt, keySize);
  addRoundKey(state, rKey, keySize);
  mixColumns(state, decrypt);
}

void aesMainInv(uchar * state, uchar * rKey, unsigned int rounds,
                  bool decrypt, unsigned int keySize)
{
  addRoundKey(state, rKey + keySize*rounds, keySize);
  for(unsigned int i=rounds-1; i > 0; --i)
  {
    aesRoundInv(state, rKey + keySize*i, decrypt, keySize);
  } 
  shiftRows(state, decrypt);
  subBytes(state, decrypt, keySize);
  addRoundKey(state, rKey, keySize);
}

/**
 *
 *
 */
void reference(uchar * output       ,
               uchar * input        ,
               uchar * rKey         ,
               unsigned int explandedKeySize,
               unsigned int width           ,
               unsigned int height          ,
               bool inverse,
               unsigned int rounds          ,
               unsigned int keySize         )
{
  uchar block[16];

  for(unsigned int blocky = 0; blocky < height/4; ++blocky)
    for(unsigned int blockx= 0; blockx < width/4; ++blockx)
    { 
      for(unsigned int i=0; i < 4; ++i)
      {
        for(unsigned int j=0; j < 4; ++j)
        {
          unsigned int index  = (((blocky * width/4) + blockx) * keySize )+ (i*4 + j);
          block[i*4 + j] = input[index];
        }
      }

      if(inverse)
        aesMainInv(block, rKey, rounds, inverse, keySize);
      else
        aesMain(block, rKey, rounds, inverse, keySize);

      for(unsigned int i=0; i <4 ; ++i)
      {
        for(unsigned int j=0; j <4; ++j)
        {
          unsigned int index  = (((blocky * width/4) + blockx) * keySize )+ (i*4 + j);
          output[index] = block[i*4 + j];
        } 
      }
    }
}


