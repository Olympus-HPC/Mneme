#include <chrono>

typedef struct atom_t {
  double pos[3] = {0,0,0};
  double eps=0; // lj
  double sig=0; // lj
  double charge=0;
  double f[3] = {0,0,0}; // force
  int molid=0;
  int frozen=0;
  double u[3] = {0,0,0}; // dipole
  double polar=0; // polarizability
} d_atom;

// function declaration
void force_kernel(
    #ifdef SYCL
    sycl::queue &q,
    #endif
    const int total_atoms,
    const int block_size,
    const int pform,
    const double cutoff,
    const double ea,
    const int kmax,
    const int kspace,
    const double pd,
    const double *h_basis,
    const double *h_rbasis,
    d_atom *h_atom_list);

void GPU_force(System &system) {

  const int N = (int)system.constants.total_atoms;
  const int block_size = system.constants.device_block_size;
  const int atoms_array_size=sizeof(d_atom)*N;
  int index=0;

  // if polarization force needed, get dipoles on CPU first
  if (system.constants.potential_form == POTENTIAL_LJESPOLAR) {
    if (system.constants.ensemble == ENSEMBLE_UVT) {
      thole_resize_matrices(system); // only if N can change
    }
    thole_amatrix(system); // populate A matrix
    thole_field(system); // calculate electric field
    int num_iterations = thole_iterative(system); // calculate dipoles
    system.stats.polar_iterations.value = (double)num_iterations;
    system.stats.polar_iterations.calcNewStats();
    system.constants.dipole_rrms = get_dipole_rrms(system);
  }

  d_atom H[N]; // host atoms
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      H[index].molid = i;
      H[index].sig = system.molecules[i].atoms[j].sig;
      H[index].eps = system.molecules[i].atoms[j].eps;
      H[index].charge = system.molecules[i].atoms[j].C;
      if (system.constants.potential_form == POTENTIAL_LJESPOLAR)
        H[index].polar = system.molecules[i].atoms[j].polar;
      for (int n=0; n<3; n++) {
        H[index].pos[n] = system.molecules[i].atoms[j].pos[n];
        H[index].f[n] = 0; // initialize to zero
        if (system.constants.potential_form == POTENTIAL_LJESPOLAR) {
          H[index].u[n] = system.molecules[i].atoms[j].dip[n];
        }
      }
      H[index].frozen = system.molecules[i].atoms[j].frozen;
      index++;
    }
  }

  int bs = sizeof(double)*9;
  double *basis = (double*)malloc(bs);
  double *reciprocal_basis = (double*)malloc(bs);

  for (int p=0; p<3; p++) {
    for (int q=0; q<3; q++) {
      basis[3*q+p] = system.pbc.basis[p][q];
      reciprocal_basis[3*q+p] = system.pbc.reciprocal_basis[p][q];
    }
  }

  // assign potential form for force calculator
  int pform,theval=system.constants.potential_form;
  if (theval == POTENTIAL_LJ || theval == POTENTIAL_LJES || theval == POTENTIAL_LJESPOLAR)
    pform=0;
  if (theval == POTENTIAL_LJES || theval == POTENTIAL_LJESPOLAR)
    pform=1;
  if (theval == POTENTIAL_LJESPOLAR)
    pform=2;

  auto start = std::chrono::steady_clock::now();

  // compute forces on a device
  force_kernel (
                #ifdef SYCL
                system.constants.q, 
                #endif
                N, 
                block_size, 
                pform, 
                system.pbc.cutoff,
                system.constants.ewald_alpha, 
                system.constants.ewald_kmax, 
                system.constants.kspace_option,
                system.constants.polar_damp,
                basis, 
                reciprocal_basis,
                H); 

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offload time: %f (s)\n", time * 1e-9f);

  index=0;
  for (int i=0; i<system.molecules.size(); i++) {
    for (int j=0; j<system.molecules[i].atoms.size(); j++) {
      for (int n=0; n<3; n++) {
        system.molecules[i].atoms[j].force[n] = H[index].f[n];
      }
      index++;
    }
  }

  free(basis);
  free(reciprocal_basis);
}

