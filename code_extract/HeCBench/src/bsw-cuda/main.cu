#include "utils.hpp"
#include "driver.hpp"

void proteinSampleRun(std::string refFile, std::string queFile, std::string out_file) {
  std::vector<std::string> G_sequencesA, G_sequencesB;
  std::string myInLine;
  unsigned long largestA = 0, largestB = 0, totSizeA = 0, totSizeB = 0;
  std::ifstream ref_file(refFile), quer_file(queFile);

  if (ref_file.is_open())
  {
    while (getline(ref_file, myInLine))
    {
      if (myInLine[0] == '>') {
        continue;
      } else {
        std::string seq = myInLine;
        G_sequencesA.push_back(seq);
        //std::cout<<"ref:"<<G_sequencesA.size()<<std::endl;
        totSizeA += seq.size();
        if(seq.size() > largestA)
        {
          largestA = seq.size();
        }
      }
    }
    ref_file.close();
  } else {
    std::cerr << "Error opening the reference file "
              << refFile << std::endl;
    return;
  }


  if (quer_file.is_open())
  {
    while (getline(quer_file, myInLine))
    {
      if (myInLine[0] == '>') {
        continue;
      } else {
        std::string seq = myInLine;
        G_sequencesB.push_back(seq);
        //std::cout<<"qeu:"<<G_sequencesB.size()<<std::endl;
        totSizeB += seq.size();
        if(seq.size() > largestB)
        {
          largestB = seq.size();
        }
      }
    }
    quer_file.close();
  } else {
    std::cerr << "Error opening the query file "
              << queFile << std::endl;
    return;
  }

  short scores_matrix[] = {
    4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,-4,
    -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,-4,
    -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,-4,
    -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,-4,
    0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
    -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,-4,
    -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4,
    0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,-4,
    -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,-4,
    -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,-4,
    -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,-4,
    -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,-4,
    -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,-4,
    -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,-4,
    -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
    1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,-4,
    0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,-4,
    -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,-4,
    -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,-4,
    0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,-4,
    -2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,-4,
    -1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4,
    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,-4,
    -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1}; // blosum 62

  kernel_driver_aa(out_file, G_sequencesB, G_sequencesA, scores_matrix, -6, -1);
}

int main(int argc, char* argv[])
{
  proteinSampleRun(argv[1], argv[2], argv[3]);
  return 0;
}
