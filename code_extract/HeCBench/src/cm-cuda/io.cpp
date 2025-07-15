#include "utils.h"

//-----------------------------------------------------------------------------------
// USER INPUT FUNCTIONS
//-----------------------------------------------------------------------------------

/*
 * Prompt the user to input the required number of random gene signatures used to assess significance
 */
int requestNRandomGenerations() {
  std::cout << "Please enter the number of random generations required over " << minRandomGenerations << " and under " << maxRandomGenerations << " (default " << defaultRandomGenerations << "): ";
  int nRandomGenerations = defaultRandomGenerations;
  std::string inputRG = "";
  getline(std::cin, inputRG);
  std::stringstream StreamRG(inputRG);
  if (StreamRG >> nRandomGenerations && !(StreamRG >> inputRG)) {
    if(nRandomGenerations > maxRandomGenerations || nRandomGenerations < minRandomGenerations) {
      nRandomGenerations = defaultRandomGenerations;
      std::cout << "Random generations choice out of range - will default to " << nRandomGenerations << std::endl;
    }
    std::cin.clear();
  }
  else
    nRandomGenerations = defaultRandomGenerations;
  return nRandomGenerations;
}


/*
 * Prompt the user to get the required expected number of false positives
 */
int requestENFP() {
  std::string inputENFP = "";
  std::cout << "Please enter the number expected number of false positives (ENFP) required, limited to "<< maxENFP << " (default " << defaultENFP << ") :";
  int ENFPvalue = defaultENFP;
  getline(std::cin, inputENFP);
  std::stringstream StreamE(inputENFP);
  if (StreamE >> ENFPvalue  && !(StreamE >> inputENFP)) {
    if (ENFPvalue > maxENFP || ENFPvalue < 1) {
      ENFPvalue = defaultENFP;
      std::cout << "ENFP choice out of range - will default to " << ENFPvalue << std::endl;
    }
    std::cin.clear();
  }
  else
    ENFPvalue = defaultENFP;
  return ENFPvalue;
}


/*
 * Prompt the user to select whether to select compounds by name with/without dose and cell line etc.
 */
int requestCompoundChoice() {
  std::cout << "Compound choice 1: Compound" << std::endl;
  std::cout << "Compound choice 2: Compound and dose" << std::endl;
  std::cout << "Compound choice 3: Compound, dose and cell line" << std::endl;
  std::cout << "Compound choice 4: Each profile is individually analysed" << std::endl;
  std::cout << "Please enter the number the compound choice (1-4) (default 1) :";
  std::string inputCC;
  int compoundChoice = defaultCompoundChoice;
  getline(std::cin, inputCC);
  std::stringstream StreamCC(inputCC);
  if (StreamCC >> compoundChoice && !(StreamCC >> compoundChoice)) {
    if(compoundChoice > 4 || compoundChoice < 1) {
      compoundChoice = defaultCompoundChoice;
      std::cout << "Compound choice out of range - will default to " << compoundChoice << std::endl;
    }
    std::cin.clear();
  }
  else
    compoundChoice = defaultCompoundChoice;
  return compoundChoice;
}


//-----------------------------------------------------------------------------------
// GENERAL FILE & DIRECTORY FUNCTIONS
//-----------------------------------------------------------------------------------

/*
 * Change the current working directory
 */
int changeToDirectory(const std::string &newPath) {
  return chdir(newPath.c_str());
}
/*
 * Or Else take the default
 */
int getCurrentPath(char* buffer) {
  getcwd(buffer, FILENAME_MAX);
  return 0;
}

/*
 * Change the current working directory to some specified sub-directory off the base
 */
int changeToSubDirectory(const std::string &basePath, const std::string &subDir) {
  int success = changeToDirectory(basePath);
  if (success == 0)
    return changeToDirectory(subDir);
  else
    return success;
}

/*
 * Populate a vector containing the names of all the files in a directory that have extensions likely to be useful
 */
int getFilesInDirectory(const std::string &dir, std::vector<std::string> &files, const std::string &extRequired){
  DIR *inputDir;
  struct dirent *currentFile;
  // Try to open the directory
  if((inputDir = opendir(dir.c_str())) == NULL) {
    std::cout << "Error(" << errno << ") opening " << dir << std::endl;
    return errno;
  }
  // Loop through the files in the directory
  while ((currentFile = readdir(inputDir)) != NULL) {
    // Get the file name
    std::string fName(currentFile->d_name);
    // We want to ignore current/previous directories, and hidden files starting with '.'
    if (fName.size() < extRequired.size() || fName[0] == '.')
      continue;
    // Extract the final 4 characters
    std::string ext = fName.substr(fName.size() - extRequired.size());
    // If we have a valid extension, add the file name to the vector
    if (ext.compare(extRequired) == 0)
      files.push_back(fName);
  }
  closedir(inputDir);
  // Sort the file names
  std::sort(files.begin(), files.end());
  return 0;
}


//-----------------------------------------------------------------------------------
// FILE READING / PARSING FUNCTIONS
//-----------------------------------------------------------------------------------

/*
 * Read and decipher a query file, populating the geneNameList and regulationValue vectors (which should have the same length)
 */
void parseQueryFile(const std::string &queries, std::vector<std::string> &geneNameList, std::vector<int> &regulationValue) {
  std::string line;
  std::ifstream inp;
  inp.open(queries.c_str(), std::ifstream::in);
  while (getline(inp, line)) {
    if (line.find("#") == 0  || line.find("AffyProbeSetID") == 0 ||  line.find("uniqueID") == 0)
      continue;
    std::stringstream ss(line);
    std::string str;
    int regulationNum = 0;
    while(ss >> str >> regulationNum) {
      regulationValue.push_back(regulationNum);
      geneNameList.push_back(str);
    }
  }
  inp.close();
}

/*
 * Parse a nice string for a drug from a file name
 */
std::string parseDrugInfoFromFilename(const std::string &fName, const int &compoundChoice) {
  std::stringstream liness(fName);
  std::string drug, dose, cellLine, cellNo;
  getline(liness, drug, separator);
  getline(liness, dose, separator);
  getline(liness, cellLine,separator);
  getline(liness, cellNo, '.');
  switch (compoundChoice) {
    case 2:
      return drug + "_" + dose;
    case 3:
      return drug + "_" + dose + '_' + cellLine;
    case 4:
      return drug + "_" + dose + '_' + cellLine + '_' + cellNo;
    default:
      return drug;
  }
}

/*
 * Extract the names for all genes within a single .tab file
 */
void populateRefGeneNameList(const std::string &fName, std::vector<std::string> &refGeneNameList) {
  std::ifstream inFile(fName.c_str());
  // Skip the first line
  std::string tmp;
  std::getline(inFile, tmp);
  // Ensure the vector is cleared
  refGeneNameList.clear();
  // Read and parse the file contents
  for (int j = 0; j < U133AArrayLength; j++) {
    // Read the next line into a temporary buffer
    std::getline(inFile, tmp);
    // Find the location of the first tab character
    int tab = tmp.find_first_of("\t");
    // Extract the first column as everything in the line up to the first tab
    refGeneNameList.push_back(tmp.substr(0, tab));
  }
  inFile.close();
}


/*
 * Extract the regulation values for all genes within a single .tab file -
 * optionally adding them to the existing values, otherwise just replacing anything that was there
 */
void populateRefRegulationValues(const std::string &fName, int *refRegNum, const bool addToCurrent = false) {
  std::ifstream inFile(fName.c_str());
  // Skip the first line
  std::string tmp;
  std::getline(inFile, tmp);
  // Read and parse the file contents
  for (int j = 0; j < U133AArrayLength; j++) {
    // Read the next line into a temporary buffer
    std::getline(inFile, tmp);
    // Find the location of the first tab character
    int tab = tmp.find_first_of("\t");
    // Extract an integer from everything after the first tab character, putting the result into the output array
    // (either adding to what's there or not)
    if (addToCurrent)
      refRegNum[j] += std::atoi(tmp.substr(tab).c_str());
    else
      refRegNum[j] = std::atoi(tmp.substr(tab).c_str());
  }
  inFile.close();
}


//-----------------------------------------------------------------------------------
// OUTPUT FUNCTION
//-----------------------------------------------------------------------------------

/*
 * Write the header of the output file, giving some info about the results
 */
void writeOutputFileHeader(std::ofstream &outdata, const std::string &sigFilename, const int &randomGenerations, const std::vector<std::string> &geneNameList, const std::vector<int> &regNum) {
  // Provide some meaningful output
  outdata << "This is the output file for signature: \t " << sigFilename << std::endl;
  outdata << "The number of random generations to generate the P-value: \t " << randomGenerations << std::endl;
  outdata << "The array size is for U133A :\t "<< U133AArrayLength << std::endl;
  outdata << std::endl;
  outdata << "The signature input was: "<< sigFilename << std::endl;
  outdata << "The signature length was: "<< geneNameList.size() << std::endl;

  // Record information about the query
  for (size_t printSig = 0; printSig < geneNameList.size(); printSig++)
    outdata << geneNameList[printSig] << "\t "<< regNum[printSig] << std::endl;
  outdata << std::endl;
}


