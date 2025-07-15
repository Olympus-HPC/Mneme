#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "kernels.h"

#define OTHER_ERROR if (cudaPeekAtLastError())printf(" error %s \n", cudaGetErrorString(cudaPeekAtLastError()))

#define HANDLE_ERROR(ans) (handleError((ans), __FILE__, __LINE__))
inline void handleError(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

#define HANDLE_KERNEL_ERROR(...) \
  __VA_ARGS__;                 \
  HANDLE_ERROR( cudaPeekAtLastError() );

// Copies the flow velocity from GPU to CPU memory, for data output.
void u_read(lbm_vars *h_vars, lbm_vars *d_vars, int nl) {
  HANDLE_ERROR(cudaMemcpy(h_vars->u_star.u0, d_vars->u_star.u0, sizeof(u_type)*nl, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_vars->u_star.u1, d_vars->u_star.u1, sizeof(u_type)*nl, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(h_vars->u_star.u2, d_vars->u_star.u2, sizeof(u_type)*nl, cudaMemcpyDeviceToHost));
}

void lbm_u_alloc(lbm_u* u, size_t nl) {
  u->u0 = (u_type*)malloc(sizeof(u_type)*nl*3);
  u->u1 = u->u0 + nl;
  u->u2 = u->u1 + nl;
}

void lbm_u_gpu_alloc(lbm_u* u, size_t nl) {
  HANDLE_ERROR(cudaMalloc(&u->u0, sizeof(u_type)*nl*3));
  u->u1 = u->u0 + nl;
  u->u2 = u->u1 + nl;
  HANDLE_ERROR(cudaMemset(u->u0, 0, sizeof(u_type)*nl*3));
}

void lbm_vars_alloc(lbm_vars* vars, lattice_type nb_dir, int nl, int vnl){

  lbm_u_alloc(&vars->u, vnl);
  lbm_u_alloc(&vars->u_star, vnl);
  lbm_u_alloc(&vars->g, vnl);
  vars->f0 = (double*)malloc(nl*nb_dir*sizeof(double));
  vars->f1 = (double*)malloc(nl*nb_dir*sizeof(double));
}

void lbm_vars_gpu_alloc(lbm_vars* vars, lattice_type nb_dir, int nl, int vnl) {
  lbm_u_gpu_alloc(&vars->u_star, vnl);
  lbm_u_gpu_alloc(&vars->g, vnl);
  HANDLE_ERROR(cudaMalloc(&vars->f0, nl*nb_dir*sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&vars->f1, nl*nb_dir*sizeof(double)));
  HANDLE_ERROR(cudaMalloc(&vars->boundary_flag, nl*sizeof(flag_type)));
  HANDLE_ERROR(cudaMalloc(&vars->boundary_values, nl*sizeof(int)));
  HANDLE_ERROR(cudaMalloc(&vars->boundary_dirs, nl*sizeof(flag_type)));
  HANDLE_ERROR(cudaMalloc(&vars->r, sizeof(double)*vnl));
}

void init_and_allocate_data(BoxCU &domain, lbm_vars *h_vars, lbm_vars *d_vars){

  const int nl = domain.nx*domain.ny*domain.nz;
  const int vnl = nl;

  HANDLE_ERROR(cudaMemcpyToSymbol(C_dirs, dirs, 57*sizeof(char), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(C_p, outer_bounds_priority, sizeof(outer_bounds_priority), 0, cudaMemcpyHostToDevice));

  // Hold all pointers needed for the lbm computation. There is one for the GPU and one for the CPU

  lbm_vars_alloc(h_vars, D3Q19, nl, vnl);
  lbm_vars_gpu_alloc(d_vars, D3Q19, nl, vnl);

  outer_wall wall{type_b::bounce, type_b::bounce, type_b::bounce, type_b::bounce, type_b::moving_wall, type_b::bounce};

  // Initialization of flags according to wall
  HANDLE_ERROR(cudaMemset(d_vars->boundary_flag, type_b::fluid,  nl*sizeof(flag_type)));

  HANDLE_KERNEL_ERROR(make_flag<<<dim3(1, domain.ny, domain.nz), domain.nx>>>(
    d_vars->boundary_flag, d_vars->boundary_values, d_vars->boundary_dirs, domain, wall, domain.nx, domain.ny, domain.nz, 0));

  HANDLE_KERNEL_ERROR(find_wall<D3Q19><<<dim3(1, domain.ny, domain.nz), domain.nx>>>(
    d_vars->boundary_flag, d_vars->boundary_dirs, d_vars->boundary_values, domain, 0));
}

void deallocate_data(lbm_vars *h_vars, lbm_vars *d_vars) {
  free(h_vars->u.u0);
  free(h_vars->u_star.u0);
  free(h_vars->g.u0);
  free(h_vars->f0);
  free(h_vars->f1);
  HANDLE_ERROR(cudaFree(d_vars->u_star.u0));
  HANDLE_ERROR(cudaFree(d_vars->g.u0));
  HANDLE_ERROR(cudaFree(d_vars->f0));
  HANDLE_ERROR(cudaFree(d_vars->f1));
  HANDLE_ERROR(cudaFree(d_vars->boundary_flag));
  HANDLE_ERROR(cudaFree(d_vars->boundary_values));
  HANDLE_ERROR(cudaFree(d_vars->boundary_dirs));
  HANDLE_ERROR(cudaFree(d_vars->r));
}

double run_benchmark(BoxCU &domain, lbm_vars h_vars, lbm_vars d_vars) {

  int N = domain.ny;

  const int bench_ini_iter = 1000;
  const int bench_max_iter = 2000;
  const int output_frame = 2000;

  const double ulb = 0.02;
  const double dx = 1. / (N - 2.);
  const double dt = dx * ulb;

  const double Re = 100.;
  const double nu = ulb * (N - 2.) / Re;
  const double omega = 1. / (3. * nu + 0.5);

  printf("omega = %f\n", omega);

  const int nl = domain.nx*domain.ny*domain.nz;

  // Initialization of the populations.
  HANDLE_KERNEL_ERROR(init_velocity_g<D3Q19><<<dim3(1, domain.ny, domain.nz), domain.nx >>>(
                      d_vars, domain, domain, domain.nz, 0, 0, 0, 1.));
  int iter = 0;
  int num_bench_iter = 0;

  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  printf("Starting %d warmup iterations\n", bench_ini_iter);
  // Main time loop of the simulation.
  for(iter = 0; iter < bench_max_iter; ++iter) {

    bool do_output = iter < bench_ini_iter && iter > 0 && (iter % output_frame == 0 || iter == 149);
    if (iter == bench_ini_iter) {
      printf("Starting %d benchmark iterations\n", bench_max_iter - bench_ini_iter);
      start = std::chrono::steady_clock::now();
    }
    if (iter >= bench_ini_iter) {
      ++num_bench_iter;
    }

    // LBM collision-straming cycle, in parallel over every cell. Stream precedes collision.
    HANDLE_KERNEL_ERROR(collide_and_stream_g<D3Q19> <<<dim3((domain.nx-1)/64+1, domain.ny, domain.nz), 64>>>(
                        d_vars, domain, ulb, omega, do_output, iter));
    // Swap populations pointer, f0 are population to be read and f1 the population to be written,
    // this is the double population soa scheme.
    double *tp = d_vars.f0;
    d_vars.f0 = d_vars.f1;
    d_vars.f1 = tp;
    cudaDeviceSynchronize();

    // Ouput average kinetic energy for validation.
    if (do_output) {
      u_read(&h_vars, &d_vars, nl);

      double energy = 0;

      for(int z = 0; z < domain.nz; ++z){
        for(int y = 0; y < domain.ny; ++y){
          for(int x = 0; x < domain.nx; ++x){
            energy += h_vars.u_star.u0[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u0[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] +
              h_vars.u_star.u1[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u1[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] +
              h_vars.u_star.u2[IDX(x,y,z, domain.nx, domain.ny, domain.nz)] *
              h_vars.u_star.u2[IDX(x,y,z, domain.nx, domain.ny, domain.nz)];
          }
        }
      }
      energy *= 0.5;

      printf("energy %f iteration %d \n", energy*dx*dx/(dt*dt), iter);

      if (iter == 149 && N == 102) {
        printf("Regression test at iteration %d: Average energy LU = %f", iter, energy);
        const double reference_energy = 2.09868507623;
        if (fabs(energy - reference_energy) < 1.e-7) {
          printf(": OK\n");
        }
        else {
          printf(": FAILED\nExpected the value %f\n", reference_energy);
        }
      }
    }
  }
  end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double mlups = ((double)nl*(double)num_bench_iter / (elapsed * 1e-9f)) / 1.e6;
  return mlups;
}

int main(int argc, char* argv[]) {
  // Uncomment the following line to select a specific GPU.
  // cudaSetDevice(1);

  if (argc != 2) {
    printf("Usage: %s <N>\n", argv[0]);
    printf("N: domain size in each dimension\n");
    return -1;
  }
  const int N = atoi(argv[1]);

  BoxCU domain;
  domain.nx = N;
  domain.ny = N;
  domain.nz = N;
  lbm_vars h_vars, d_vars;

  init_and_allocate_data(domain, &h_vars, &d_vars);

  double mlups[10];
  for (int i = 0; i < 10; ++i)
    mlups[i] = run_benchmark(domain, h_vars, d_vars);

  deallocate_data(&h_vars, &d_vars);

  printf("\nPerformance: MLUPS\n");
  for (int i = 0; i < 10; ++i) {
    printf("%.4f\n", mlups[i]);
  }

  return 0;
}
