static void error(std::string errorMsg)
{
  std::cout<<"Error: "<<errorMsg<<std::endl;
}

template<typename T>
int isPowerOf2(T val)
{
  long long _val = val;
  if((_val & (-_val))-_val == 0 && _val != 0)
    return 0;
  else
    return 1;
}

template<typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for(int i = 0; i < bytes; i++)
    val |= val >> (1<<i);
  val++;
  return val;
}


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
    error("Cannot fill array. NULL pointer.");
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


#ifdef DEBUG
template<typename T>
void printArray(
    const std::string header,
    const T * data,
    const int width,
    const int height)
{
  std::cout<<"\n"<<header<<"\n";
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      std::cout<<data[i*width+j]<<" ";
    }
    std::cout<<"\n";
  }
  std::cout<<"\n";
}
#endif

static bool compare(const float *refData, const float *data,
                    const int length, const float epsilon = 1e-6f)
{
  float error = 0.0f;
  float ref = 0.0f;
  for(int i = 1; i < length; ++i)
  {
    float diff = refData[i] - data[i];
    error += diff * diff;
    ref += refData[i] * refData[i];
  }
  float normRef =::sqrtf((float) ref);
  if (::fabs((float) ref) < 1e-7f)
  {
    return false;
  }
  float normError = ::sqrtf((float) error);
  error = normError / normRef;
  return error < epsilon;
}




#define CHECK_ALLOCATION(actual, msg) \
  if(actual == NULL) \
{ \
  error(msg); \
  std::cout << "Location : " << __FILE__ << ":" << __LINE__<< std::endl; \
  return 1; \
}


