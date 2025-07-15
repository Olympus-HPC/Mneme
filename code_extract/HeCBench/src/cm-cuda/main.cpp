#include "utils.h"

int main (int argc, char *argv[]) {

  std::string basePath;
  // An input directory is required for us to be able to do anything
  if (argc < 2) {
    std::cout << "Changing to current directory " << std::endl;
    char homePath[FILENAME_MAX];
    getCurrentPath(homePath);
    std::string basePath2(homePath);
    basePath+=basePath2;
  }

  // Read the input argument, and see if it's a directory we can move into
  else{
    std::string basePath2(argv[1]);
    basePath+=basePath2;
    if (changeToDirectory(basePath) != 0) {
      std::cout << "Could not open the directory " << basePath << std::endl;
      exit(-1);
    }
  }

  // State what is the base directory containing the files (in sub-directories) to be processed
  std::cout << "The base directory is : " << basePath << std::endl;

  // Initialize the parameters (from user input, or predefined defaults)
  int nRandomGenerations = defaultRandomGenerations;
  int ENFPvalue = defaultENFP;
  int compoundChoice = defaultCompoundChoice;
  if (promptForInput) {
    nRandomGenerations = requestNRandomGenerations();
    ENFPvalue = requestENFP();
    compoundChoice = requestCompoundChoice();
  }

  // Start to measure the processing time
  auto startTime = std::chrono::steady_clock::now();

  // Change to the queries sub-directory and get a list of the query files
  if (changeToSubDirectory(basePath, subDirQueries) != 0) {
    std::cout << "Could not find a sub-directory with the name " << subDirQueries << " within " << basePath << std::endl;
    return -1;
  }
  std::vector<std::string> filesQueries = std::vector<std::string>();
  getFilesInDirectory(".", filesQueries, ".sig");
  if (filesQueries.size() == 0) {
    std::cout << "No query files (extension .sig) found!" << std::endl;
    exit(0);
  }

  // Flag to indicate that something, at some stage, succeeded
  bool success = false;

  // Loop through the query files, processing them individually
  for (size_t queryFileNo = 0; queryFileNo < filesQueries.size(); queryFileNo++) {

    // Extract the name of the query file to be processed now
    std::string fNameQuery = filesQueries[queryFileNo];

    // Go inside the queries sub-directory
    if (changeToSubDirectory(basePath, subDirQueries) != 0) {
      std::cout << "Could not find a sub-directory with the name " << subDirQueries << " within " << basePath << std::endl;
      return -1;
    }

    // Read the signature gene names and regulation values (-1 or 1) from a query file
    std::cout << "Query filename : "<< fNameQuery << std::endl;
    std::vector<std::string> sigGeneNameList;
    std::vector<int> sigRegValue;
    parseQueryFile(fNameQuery, sigGeneNameList, sigRegValue);

    // Open a suitable output file stream for writing in the results directory
    if (changeToSubDirectory(basePath, subDirResults) != 0) {
      std::cout << "Could not find a sub-directory with the name " << subDirResults << " within " << basePath << std::endl;
      return -1;
    }
    std::string fileHeader = "sscMap_Result_[" + fNameQuery + "].tab";
    std::ofstream outputStream(fileHeader.c_str());
    if(!outputStream) {
      std::cerr << "Error: " << fileHeader << " could not be opened." << std::endl;
      exit(-1);
    }

    // Write some general information about the analysis to the output file
    outputStream << "Number of random generations: " << nRandomGenerations << std::endl;
    outputStream << "Expected number of false positives: " << ENFPvalue << std::endl;
    outputStream << "Compound choice: " << compoundChoice << std::endl;

    // Write some general information about the analysis to the output file
    writeOutputFileHeader(outputStream, fNameQuery, nRandomGenerations, sigGeneNameList, sigRegValue);

    // Populate a vector giving the file names for compounds
    if (changeToSubDirectory(basePath, subDirFiles) != 0) {
      std::cout << "Could not find a sub-directory with the name " << subDirFiles << " within " << basePath << std::endl;
      outputStream.close();
      return -1;
    }
    std::vector<std::string> refFiles = std::vector<std::string>();
    getFilesInDirectory(".", refFiles, ".tab");
    if (filesQueries.size() == 0) {
      std::cout << "No suitable files (extension .tab) found!" << std::endl;
      outputStream.close();
      exit(0);
    }

    // Process the query itself
    int querySuccess = processQuery(refFiles, sigGeneNameList, sigRegValue, nRandomGenerations, compoundChoice, ENFPvalue, outputStream);
    if (querySuccess != 0) {
      std::cout << "*** Query processing failed with error code " << querySuccess << " ***" << std::endl;
      std::cout << "Lack of memory is a probable cause - you could try again with a lower number of random values." << std::endl;
      std::cout << "Also, try closing some applications (e.g. web browsers) before trying again." << std::endl;
    } else
      success = true;

    // Done with the output stream for this query - just close it
    outputStream.close();

    // Some clean-up
    refFiles.clear();
    sigRegValue.clear();
    sigGeneNameList.clear();
  }

  // Output the total processing time
  auto endTime = std::chrono::steady_clock::now();
  auto totalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
  std::cout << "Time in taken in seconds :"<< (double)totalTime * 1e-9 << std::endl;

  // Output a friendly message, unless everything failed
  if (success)
    std::cout << "Program is finished, output files will be found in the results folder"<< std::endl;

  return 0;
}


