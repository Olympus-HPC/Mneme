#ifndef FILEREADER_H
#define FILEREADER_H

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string>

using std::string;

class FileReader {
    private:
        char* filename;
        char* ptr;
        struct stat sb;

    public:
        FileReader(char* f, void* pPos = NULL): filename(f) {
            int fd = open(filename, O_RDONLY);
            fstat(fd, &sb);
            ptr = static_cast<char*>(mmap(
                pPos, sb.st_size, PROT_READ, MAP_SHARED, fd, 0));
        }
        ~FileReader() { close(); }

        void close() {
            if (ptr != MAP_FAILED)
                munmap(ptr, sb.st_size);
        }

        inline char* getPtr     () const { return ptr; }
        inline void* getFileEnd () const { return (void*)(ptr+sb.st_size); }
        inline off_t getFileSize() const { return sb.st_size; }
};

#endif