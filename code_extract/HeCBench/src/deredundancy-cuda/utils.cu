#include "utils.h"

//--------------------function--------------------//
// printUsage
void printUsage() {
  std::cout << "Check the input please."  << std::endl;
  std::cout << "a.out i inputFile t threshold" << std::endl;
  exit(0);
}

// checkOption
void checkOption(int argc, char **argv, Option &option) {
  if (argc%2 != 1) printUsage();  // parameter shoud be odd
  option.inputFile = "testData.fasta";  // input file name
  option.outputFile = "result.fasta";  // output file name
  option.threshold = 0.95;
  for (int i=1; i<argc; i+=2) {  // parsing parameters 
    switch (argv[i][0]) {
      case 'i':
        option.inputFile = argv[i+1];
        break;
      case 'o':
        option.outputFile = argv[i+1];
        break;
      case 't':
        option.threshold = std::atof(argv[i+1]);
        break;
      default:
        printUsage();
        break;
    }
  }
  if (option.threshold < 0.8 || option.threshold >= 1) {
    std::cout << "Threshold out of range." << std::endl;
    exit(0);
  }
  int temp = (option.threshold*100-80)/5;
  switch (temp) {  // wordLength decided by threshold
    case 0:  // threshold:0.80-0.85 wordLength:4
      option.wordLength = 4;
      break;
    case 1:  // threshold:0.85-0.90 wordLength:5
      option.wordLength = 5;
      break;
    case 2:  // threshold:0.90-0.95 wordLength:6
      option.wordLength = 6;
      break;
    case 3:  // threshold:0.90-1.00 wordLength:7
      option.wordLength = 7;
      break;
  }
  std::cout << "input:\t" << option.inputFile << std::endl;
  std::cout << "output:\t" << option.outputFile << std::endl;
  std::cout << "threshold:\t" << option.threshold << std::endl;
  std::cout << "word length:\t" << option.wordLength << std::endl;
}

// compare
bool compare(const Read &a, const Read &b) {
  return a.data.size() > b.data.size();
}

// readFile
bool readFile(std::vector<Read> &reads, Option &option) {
  std::ifstream file(option.inputFile.c_str());
  if (!file.is_open()) {
    std::cout << "Failed to open file " << option.inputFile << ". Exit\n";
    return true;
  }
  Read read;
  std::string line;
  getline(file, line);
  read.name = line;
  while (getline(file, line)) {  // getline has no \n
    if (line[0] == '>') {
      reads.push_back(read);
      read.name = line;
      read.data = "";
      continue;
    }
    read.data += line;
  }
  reads.push_back(read);
  file.close();
  std::sort(reads.begin(), reads.end(), compare);
  std::cout << "reads count:\t" << reads.size() << std::endl;
  return false;
}
