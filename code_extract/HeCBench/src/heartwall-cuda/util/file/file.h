#ifdef __cplusplus
extern "C" {
#endif

void 
read_parameters(const char* filename,
                int* tSize,
                int* sSize,
                int* maxMove,
                fp* alpha);

void 
read_header(const char* filename, int* size, int* size_2);

void 
read_data(const char* filename,
          int size,
          int* input_a,
          int* input_b,
          int size_2,
          int* input_2a,
          int* input_2b);

void 
write_data(const char* filename,
           int frameNo,
           int frames_processed,
           int endoPoints,
           int* input_a,
           int* input_b,
           int epiPoints,
           int* input_2a,
           int* input_2b);

#ifdef __cplusplus
}
#endif
