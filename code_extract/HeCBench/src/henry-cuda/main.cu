#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda.h>

#define NUMTHREADS 256  // number of threads per GPU block

// data for atom of crystal structure
//    Unit cell of crystal structure can then be stored 
//    as pointer array of StructureAtom's
struct StructureAtom {
  // Cartesian position, units: A
  double x;
  double y;
  double z;
  // Lennard-Jones epsilon parameter with adsorbate
  double epsilon;  // units: K
  // Lennard-Jones sigma parameter with adsorbate
  double sigma;  // units: A
};

// temperature, Kelvin
const double T = 298.0; 

// Universal gas constant, m3 - Pa / (K - mol)
const double R = 8.314; 

// Generate a random number 
__device__
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

// Compute the Boltzmann factor of methane at point (x, y, z) inside structure
//   Loop over all atoms of unit cell of crystal structure
//   Find nearest image to methane at point (x, y, z) for application of periodic boundary conditions
//   Compute energy contribution due to this atom via the Lennard-Jones potential
__device__ double compute(double x, double y, double z,
    const StructureAtom * __restrict__ structureAtoms,
    double natoms, double L) 
{
  // (x, y, z) : Cartesian coords of methane molecule
  // structureAtoms : pointer array storing info on unit cell of crystal structure
  // natoms : number of atoms in crystal structure
  // L : box length
  // returns Boltzmann factor e^{-E/(RT)}
  double E = 0.0;  // energy (K)

  // loop over atoms in crystal structure
  for (int i = 0; i < natoms; i++) {
    // Compute distance in each coordinate from (x, y, z) to this structure atom
    double dx = x - structureAtoms[i].x;
    double dy = y - structureAtoms[i].y;
    double dz = z - structureAtoms[i].z;

    // apply nearest image convention for periodic boundary conditions
    const double boxupper = 0.5 * L;
    const double boxlower = -boxupper;

    dx = (dx >  boxupper) ? dx-L : dx;
    dx = (dx >  boxupper) ? dx-L : dx;
    dy = (dy >  boxupper) ? dy-L : dy;
    dy = (dy <= boxlower) ? dy-L : dy;
    dz = (dz <= boxlower) ? dz-L : dz;
    dz = (dz <= boxlower) ? dz-L : dz;

    // compute inverse distance
    double rinv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz);

    // Compute contribution to energy of adsorbate at (x, y, z) due to this atom
    // Lennard-Jones potential (this is the efficient way to compute it)
    double sig_ovr_r = rinv * structureAtoms[i].sigma;
    double sig_ovr_r6 = pow(sig_ovr_r, 6);
    double sig_ovr_r12 = sig_ovr_r6 * sig_ovr_r6;
    E += 4.0 * structureAtoms[i].epsilon * (sig_ovr_r12 - sig_ovr_r6);
  }
  return exp(-E / (R * T));  // return Boltzmann factor
}

// Inserts a methane molecule at a random position inside the structure
// Calls function to compute Boltzmann factor at this point
// Stores Boltzmann factor computed at this thread in boltzmannFactors
__global__ void insertions(
    double *__restrict__ boltzmannFactors, 
    const StructureAtom * __restrict__ structureAtoms, 
    int natoms, double L) 
{
  // boltzmannFactors : pointer array in which to store computed Boltzmann factors
  // structureAtoms : pointer array storing atoms in unit cell of crystal structure
  // natoms : number of atoms in crystal structure
  // L : box length
  int id = threadIdx.x + blockIdx.x * NUMTHREADS;

  // random seed for each thread
  uint64_t seed = id;

  // Generate random position inside the cubic unit cell of the structure
  double x = L * LCG_random_double(&seed);
  double y = L * LCG_random_double(&seed);
  double z = L * LCG_random_double(&seed);

  // Compute Boltzmann factor, store in boltzmannFactors
  boltzmannFactors[id] = compute(x, y, z, structureAtoms, natoms, L);
}

int main(int argc, char *argv[]) {
  // take in number of MC insertions as argument
  if (argc != 3) {
    printf("Usage: ./%s <material file> <ninsertions>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Import unit cell of nanoporous material IRMOF-1
  StructureAtom *structureAtoms;  // store atoms in pointer array here
  // open crystal structure file
  std::ifstream materialfile(argv[1]);
  if (materialfile.fail()) {
    printf("Failed to import file %s.\n", argv[1]);
    exit(EXIT_FAILURE);
  }

  const int ncycles = atoi(argv[2]);  // Number of Monte Carlo insertions

  // Energetic model for interactions of methane molecule with atoms of framework
  //    pairwise Lennard-Jones potentials

  // Epsilon parameters for Lennard-Jones potential (K)
  std::map<std::string, double> epsilons;
  epsilons["Zn"] = 96.152688;
  epsilons["O"] = 66.884614;
  epsilons["C"] = 88.480032;
  epsilons["H"] = 57.276566;

  // Sigma parameters for Lennard-Jones potential (A)
  std::map<std::string, double> sigmas;
  sigmas["Zn"] = 3.095775;
  sigmas["O"] = 3.424075;
  sigmas["C"] = 3.580425;
  sigmas["H"] = 3.150565;

  // read cubic box dimensions
  std::string line;
  getline(materialfile, line);
  std::istringstream istream(line);

  double L;  // dimension of cube
  istream >> L;
  printf("L = %f\n", L);

  // waste line
  getline(materialfile, line);

  // get number of atoms of a material
  getline(materialfile, line);
  int natoms;  // e.g. number of atoms in unit cell of IRMOF-1
  istream.str(line);
  istream.clear();
  istream >> natoms;
  printf("%d atoms\n", natoms);

  // waste line
  getline(materialfile, line);

  // Allocate space for material atoms and epsilons/sigmas 
  structureAtoms = (StructureAtom *) malloc(natoms * sizeof(StructureAtom));

  // read atom coordinates
  for (int i = 0; i < natoms; i++) {
    // read atoms from .cssr file
    getline(materialfile, line);
    istream.str(line);
    istream.clear();

    int atomno;
    double xf, yf, zf;  // fractional coordinates
    std::string element;

    istream >> atomno >> element >> xf >> yf >> zf;

    // load structureAtoms with Cartesian coordinates of this atom
    structureAtoms[i].x = L * xf;
    structureAtoms[i].y = L * yf;
    structureAtoms[i].z = L * zf;

    // store epsilon and sigma Lennard-Jones parameters as well, for ease of computation on device
    structureAtoms[i].epsilon = epsilons[element];
    structureAtoms[i].sigma = sigmas[element];
  }

  // Allocate space for storing atoms
  StructureAtom *d_structureAtoms;
  cudaMalloc((void**)&d_structureAtoms, natoms * sizeof(StructureAtom));
  cudaMemcpy(d_structureAtoms, structureAtoms, natoms * sizeof(StructureAtom), cudaMemcpyHostToDevice);

  // calculate number of MC insertions
  const int nBlocks = 1024;
  const int insertionsPerCycle = nBlocks * NUMTHREADS;
  const int ninsertions = ncycles * insertionsPerCycle;  

  // Allocate space for storing Boltzmann factors computed on each thread
  double * d_boltzmannFactors;
  cudaMalloc((void**)&d_boltzmannFactors, insertionsPerCycle * sizeof(double));

  double * boltzmannFactors = (double*) malloc (insertionsPerCycle * sizeof(double));

  //  Compute the Henry coefficient
  //  KH = < e^{-E/(kB * T)} > / (R * T)
  //  Brackets denote average over space

  double total_time = 0.0;

  double KH = 0.0;  // will be Henry coefficient
  for (int cycle = 0; cycle < ncycles; cycle++) {

    auto start = std::chrono::steady_clock::now();

    //  Perform Monte Carlo insertions in parallel on the GPU
    insertions<<<nBlocks, NUMTHREADS>>>(d_boltzmannFactors, d_structureAtoms, natoms, L);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    cudaMemcpy(boltzmannFactors, d_boltzmannFactors, insertionsPerCycle * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute Henry coefficient from the sampled Boltzmann factors
    for(int i = 0; i < insertionsPerCycle; i++)
      KH += boltzmannFactors[i];
  }

  // average Boltzmann factor: < e^{-E/(kB/T)} >
  KH = KH / ninsertions;  
  KH = KH / (R * T);  // divide by RT
  printf("Used %d blocks with %d thread each\n", nBlocks, NUMTHREADS);
  printf("Henry constant = %e mol/(m3 - Pa)\n", KH);
  printf("Number of actual insertions: %d\n", ninsertions);
  printf("Number of times we called the device kernel: %d\n", ncycles);
  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9) / ncycles);

  cudaFree(d_structureAtoms);
  cudaFree(d_boltzmannFactors);
  free(structureAtoms);
  free(boltzmannFactors);
  return EXIT_SUCCESS;
}
