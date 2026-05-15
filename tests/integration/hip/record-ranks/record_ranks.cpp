// Integration test: verifies MNEME_RECORD_RANKS gates per-rank recording.
//
// Each rank sets MNEME_DATA_DIR to <base>/rank-<rank> before any GPU work,
// launches a small kernel a handful of times, then on rank 0 inspects every
// rank's data dir for JSON files and asserts the ranks that produced JSONs
// match the expected set passed by the test driver.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mpi.h>
#include <set>
#include <sstream>
#include <string>

#define gpuErrCheck(CALL)                                                      \
  {                                                                            \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      printf("ERROR @ %s:%d ->  %s\n", __FILE__, __LINE__,                     \
             hipGetErrorString(err));                                          \
      std::abort();                                                            \
    }                                                                          \
  }

__global__ void smallKernel(int *Out, int Mode) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  Out[idx] = idx + Mode;
}

namespace {

std::set<int> expectedRecordingRanks(const char *Spec) {
  std::set<int> Out;
  std::stringstream SS(Spec);
  std::string Tok;
  while (std::getline(SS, Tok, ','))
    if (!Tok.empty())
      Out.insert(std::stoi(Tok));
  return Out;
}

int countJsonFiles(const std::filesystem::path &Dir) {
  if (!std::filesystem::exists(Dir))
    return 0;
  int N = 0;
  for (const auto &Entry : std::filesystem::directory_iterator(Dir))
    if (Entry.path().extension() == ".json")
      ++N;
  return N;
}

} // namespace

int main(int argc, char **argv) {
  int Provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &Provided);

  int Rank = 0;
  int Size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Size);

  if (argc != 2) {
    if (Rank == 0)
      std::cerr << "FAIL: expected comma-separated recording ranks argument\n";
    MPI_Finalize();
    return 1;
  }

  const char *DataDirBase = std::getenv("MNEME_DATA_DIR_BASE");
  if (!DataDirBase) {
    if (Rank == 0)
      std::cerr << "FAIL: MNEME_DATA_DIR_BASE must be set\n";
    MPI_Finalize();
    return 1;
  }

  // Per-rank data dir, set BEFORE any GPU API call so the recorder picks it
  // up on first interception.
  std::string PerRank =
      std::string(DataDirBase) + "/rank-" + std::to_string(Rank);
  std::filesystem::create_directories(PerRank);
  setenv("MNEME_DATA_DIR", PerRank.c_str(), 1);

  // Provide rank info to the recorder via the standard env vars in case the
  // launcher only set its own (e.g. FLUX_TASK_RANK without
  // OMPI_COMM_WORLD_RANK).
  setenv("OMPI_COMM_WORLD_RANK", std::to_string(Rank).c_str(), 1);
  setenv("OMPI_COMM_WORLD_SIZE", std::to_string(Size).c_str(), 1);

  // Tiny GPU workload — enough to produce a JSON when recording is enabled.
  constexpr int N = 64;
  int *Buf = nullptr;
  gpuErrCheck(hipMalloc(&Buf, N * sizeof(int)));
  for (int Mode = 0; Mode < 3; ++Mode) {
    smallKernel<<<dim3(1), dim3(N)>>>(Buf, Mode);
    gpuErrCheck(hipDeviceSynchronize());
  }
  gpuErrCheck(hipFree(Buf));

  MPI_Barrier(MPI_COMM_WORLD);

  int ExitCode = 0;
  if (Rank == 0) {
    auto Expected = expectedRecordingRanks(argv[1]);
    std::cout << "Size=" << Size << " expected recording ranks={";
    bool First = true;
    for (int R : Expected) {
      if (!First)
        std::cout << ",";
      std::cout << R;
      First = false;
    }
    std::cout << "}\n";

    for (int R = 0; R < Size; ++R) {
      auto Dir =
          std::filesystem::path(DataDirBase) / ("rank-" + std::to_string(R));
      int NumJson = countJsonFiles(Dir);
      bool ShouldRecord = Expected.count(R) > 0;
      std::cout << "rank " << R << " dir=" << Dir.string()
                << " json_count=" << NumJson
                << " expect_recording=" << (ShouldRecord ? "yes" : "no")
                << "\n";
      if (ShouldRecord && NumJson == 0) {
        std::cerr << "FAIL: rank " << R
                  << " was expected to record but produced 0 JSONs\n";
        ExitCode = 1;
      } else if (!ShouldRecord && NumJson != 0) {
        std::cerr << "FAIL: rank " << R
                  << " was expected to skip recording but produced " << NumJson
                  << " JSONs\n";
        ExitCode = 1;
      }
    }
    if (ExitCode == 0)
      std::cout << "PASS: per-rank JSON counts match expected recording set\n";
  }

  MPI_Finalize();
  return ExitCode;
}
