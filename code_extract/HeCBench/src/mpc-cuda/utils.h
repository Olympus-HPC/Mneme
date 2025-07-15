static long* readFile(const char name[], int &length)
{
  FILE *f = fopen(name, "rb");  assert(f != NULL);
  fseek(f, 0, SEEK_END);
  long long size = ftell(f);  assert(size > 0);
  assert(size <= 2082408380);
  assert((size % sizeof(long)) == 0);
  size /= sizeof(long);
  long* input = new long[size];
  fseek(f, 0, SEEK_SET);
  length = fread(input, sizeof(long), size, f);  assert(length == size);
  fclose(f);
  return input;
}

void writeFile (const char* filename, const long* output, const int outsize) {
  FILE *f = fopen(filename, "wb");
  if (f != NULL) {
    int length = fwrite(output, sizeof(long), outsize, f);
    if (length != outsize) {
      printf("Please check output file %s for errors\n", filename);
    }
    fclose(f);
  }
}

