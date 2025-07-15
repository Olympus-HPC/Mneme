//
// utilities called by functions in main.cpp
//
void convertColorToGray(const uchar4 *pixels, 
                        uchar *gray,
                        const int height,
                        const int width)
{
  for(int i=0; i< height; ++i)
    for(int j=0; j<width; ++j)
    {
      unsigned int index = i*width + j;
      // gray = (0.3*R + 0.59*G + 0.11*B)
      gray[index] = (uchar) (pixels[index].x * 0.3f  + 
                             pixels[index].y * 0.59f + 
                             pixels[index].z * 0.11f );
    }
}

void convertGrayToGray(const uchar4 *pixels, 
                       uchar *gray,
                       const int height,
                       const int width)
{
  for(int i=0; i< height; ++i)
    for(int j=0; j<width; ++j)
    {
      unsigned int index = i*width + j;
      gray[index] = pixels[index].x;
    }
}

void createRoundKey(uchar * eKey, uchar * rKey)
{
  for(unsigned int i=0; i < 4; ++i)
    for(unsigned int j=0; j < 4; ++j)
    {
      rKey[i+ j*4] = eKey[i*4 + j];
    }
}

void rotate(uchar * word)
{
  uchar c = word[0];
  for(unsigned int i=0; i<3; ++i)
    word[i] = word[i+1];
  word[3] = c;
}

void core(uchar * word, unsigned int iter)
{
  rotate(word);

  for(unsigned int i=0; i < 4; ++i)
  {
    word[i] = getSBoxValue(word[i]);
  }    

  word[0] = word[0]^getRconValue(iter);
}


void keyExpansion(uchar * key, uchar * expandedKey,
                  unsigned int keySize, unsigned int explandedKeySize)
{
  unsigned int currentSize   = 0;
  unsigned int rConIteration = 1;
  uchar temp[4] = {0};

  for(unsigned int i=0; i < keySize; ++i)
  {
    expandedKey[i] = key[i];
  }

  currentSize += keySize;

  while(currentSize < explandedKeySize)
  {
    for(unsigned int i=0; i < 4; ++i)
    {
      temp[i] = expandedKey[(currentSize - 4) + i];
    }

    if(currentSize%keySize == 0)
    {
      core(temp, rConIteration++);
    }

    //XXX: add extra SBOX here if the keySize is 32 Bytes

    for(unsigned int i=0; i < 4; ++i)
    {
      expandedKey[currentSize] = expandedKey[currentSize - keySize]^temp[i];
      currentSize++;
    }
  }
}

/**
 * fill array with random values
 */
template<typename T>
int fillRandom(
    T * arrayPtr,
    const int width,
    const int height,
    const T rangeMin,
    const T rangeMax,
    unsigned int seed=123)
{
  if(!arrayPtr)
  {
    std::cerr << "Cannot fill array. NULL pointer.\n";
    return 1;
  }
  if(!seed)
  {
    seed = (unsigned int)time(NULL);
  }
  srand(seed);
  double range = double(rangeMax - rangeMin) + 1.0;
  /* random initialisation of input */
  for(int i = 0; i < height; i++)
    for(int j = 0; j < width; j++)
    {
      int index = i*width + j;
      arrayPtr[index] = rangeMin + T(range*rand()/(RAND_MAX + 1.0));
    }
  return 0;
}
