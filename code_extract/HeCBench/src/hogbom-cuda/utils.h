std::vector<float> readImage(const std::string& filename)
{
  struct stat results;
  if (stat(filename.c_str(), &results) != 0) {
    std::cerr << "Error: Could not stat " << filename << std::endl;
    exit(1);
  }

  std::vector<float> image(results.st_size / sizeof(float));
  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
  file.read(reinterpret_cast<char *>(&image[0]), results.st_size);
  file.close();
  return image;
}

#ifdef OUTPUT
void writeImage(const std::string& filename, std::vector<float>& image)
{
  std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<char *>(&image[0]), image.size() * sizeof(float));
  file.close();
}
#endif

size_t checkSquare(std::vector<float>& vec)
{
  const size_t size = vec.size();
  const size_t singleDim = sqrt(size);
  if (singleDim * singleDim != size) {
    std::cerr << "Error: Image is not square" << std::endl;
    exit(1);
  }

  return singleDim;
}

void zeroInit(std::vector<float>& vec)
{
  for (std::vector<float>::size_type i = 0; i < vec.size(); ++i) {
    vec[i] = 0.0;
  }
}

bool compare(const std::vector<float>& expected, const std::vector<float>& actual)
{
  if (expected.size() != actual.size()) {
    std::cout << "Fail (Vector sizes differ)" << std::endl;
    return false;
  }

  const size_t len = expected.size();
  for (size_t i = 0; i < len; ++i) {
    if (fabs(expected[i] - actual[i]) > 1e-3) {
      std::cout << "Fail (Expected " << expected[i] << " got "
        << actual[i] << " at index " << i << ")" << std::endl;
      return false;
    }
  }

  return true;
}

