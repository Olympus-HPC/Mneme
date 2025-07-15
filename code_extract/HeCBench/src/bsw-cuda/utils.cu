#include "utils.hpp"
unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}

int get_new_min_length(short* alAend, short* alBend, int blocksLaunched){
  int newMin = 1000;
  int maxA = 0;
  int maxB = 0;
  for(int i = 0; i < blocksLaunched; i++){
    if(alBend[i] > maxB ){
      maxB = alBend[i];
    }
    if(alAend[i] > maxA){
      maxA = alAend[i];
    }
  }
  newMin = (maxB > maxA)? maxA : maxB;
  return newMin;
}


