
#define IDX(x, y, z, nx, ny, nz)    ((x + (nx))%(nx) + ((y + (ny))%(ny) + ((z + (nz))%(nz))*(ny))*(nx))
#define IDF(x, y, z, f, nx, ny, nz) ((x + (nx))%(nx) + ((y + (ny))%(ny) + ((z + (nz))%(nz))*(ny))*(nx)) + (f)*(nx)*(ny)*(nz)
#define IBAR(i,nbdir) ((((i) + (nbdir)/2 - 1)%((nbdir) - 1) + 1))

#define SET_FLAG(maybe_flag, side_of_domain)\
  if (maybe_flag >= zou_he) {\
    dir[IDX(x, y, z, nx, ny, nz)] = side_of_domain;\
  }\
  if (C_p[flags[IDX(x, y, z, nx, ny, nz)]] < C_p[maybe_flag]) { \
    flags[IDX(x, y, z, nx, ny, nz)] = maybe_flag;  \
    if(maybe_flag == type_b::bounce)bounce_val[IDX(x, y, z, nx, ny, nz)] = type_b::bounce;\
  } else if ( (maybe_flag == zou_he || flags[IDX(x, y, z, nx, ny, nz)] == zou_he || maybe_flag == zou_he_p || flags[IDX(x, y, z, nx, ny, nz)] == zou_he_p)&&flags[IDX(x, y, z, nx, ny, nz)] != bounce) { \
    flags[IDX(x, y, z, nx, ny, nz)] = corner;\
  }

enum type_b { fluid, nothing, wall_m, wall, bounce, free_slip, neumann, zou_he, zou_he_p, corner, moving_wall, equilibre};
enum dir {x, x_, y, y_, z, z_};
enum stream_type { normal, tao_stream };
enum lattice_type{ D3Q19 = 19, D3Q27 = 27};

typedef char flag_type;
typedef float u_type;

struct BoxCU {
  int x0 = 0, nx = 0, y0 = 0, ny = 0, z0 = 0, nz = 0;
};

struct lbm_u {
  u_type* u0;
  u_type* u1;
  u_type* u2;
};

typedef struct {
  lbm_u u;
  lbm_u u_star;
  lbm_u g;
  double *r;
  double *f0;
  double *f1;
  flag_type *boundary_flag;
  int *boundary_values;
  flag_type *boundary_dirs;
} lbm_vars;

// Metadata used to describe the type of outer boundary to be applied to each corresponding wall
typedef struct outer_wall {
  union {
    struct {
      int xmin, xmax, ymin, ymax, zmin, zmax;
    };
    int el[6];
  };
} outer_wall;

// The discrete velocity vectors for the D3Q19 lattice
// Warning: these constants aren't actually used in the GPU computation, which
//          are unrolled for performance reason, so modifying this array won't
//          impact the actual computation.
static const char dirs[] = {
  0, 0, 0,  //  0
  -1, 0, 0, //  1
  0,-1, 0,  //  2
  0, 0,-1,  //  3
  -1,-1, 0, //  4
  -1, 1, 0, //  5
  -1, 0,-1, //  6
  -1, 0, 1, //  7
  0,-1,-1,  //  8
  0,-1, 1,  //  9
  1, 0, 0,  // 10
  0, 1, 0,  // 11
  0, 0, 1,  // 12
  1, 1, 0,  // 13
  1,-1, 0,  // 14
  1, 0, 1,  // 15
  1, 0,-1,  // 16
  0, 1, 1,  // 17
  0, 1,-1   // 18
};

__constant__ char C_dirs[81];

static char outer_bounds_priority[12] {
    0,  // fluid,
    1,  // nothing
    2,  // wall,
    3,  // wall_m,
    7,  // bounce,
    6,  // free_slip,
    11, // neumann,
    4,  // zou_he,
    5,  // zou_he_p,
    9,  // corner,
    10, // moving_wall,
    8,  // equilibre
};

__constant__ static char C_p[12];
// Defines flags for domain boundary condition, only used at initilization
__global__ void make_flag(
  flag_type *__restrict__ flags,
        int *__restrict__ bounce_val,
  flag_type *__restrict__ dir,
  BoxCU domain,
  outer_wall wall_type,
  int width, int height, int depth,
  int iter)
{
  int nx = domain.nx;
  int ny = domain.ny;
  int nz = domain.nz;

  for (int z = blockIdx.z; z < nz; z += gridDim.z) {
    for (int y = threadIdx.y + blockIdx.y*blockDim.y; y < ny; y += blockDim.y*gridDim.y) {
      for (int x = threadIdx.x + blockIdx.x*blockDim.x; x < nx; x += blockDim.x*gridDim.x) {

        if (x + domain.x0 == 0 ) {
          SET_FLAG(wall_type.xmin, dir::x)
        }else if (x + domain.x0 == width - 1) {
          SET_FLAG(wall_type.xmax, dir::x_)
        }

        if (y + domain.y0 == 0 ) {
          SET_FLAG(wall_type.ymin, dir::y)
        }else if (y + domain.y0 == height - 1) {
          SET_FLAG(wall_type.ymax, dir::y_)
        }

        if (z + domain.z0 == 0 ) {
          SET_FLAG(wall_type.zmin, dir::z)
        }else if (z + domain.z0 == depth - 1) {
          SET_FLAG(wall_type.zmax, dir::z_)
        }
      }
    }
  }
}

// Defines flags for domain boundary conditions. Finds the lattice cells exactly at the limit of a
// bounce back obstacle and flags them as "wall", only used at initilization.
template<int nb_directions>
__global__ void find_wall(
  flag_type *__restrict__ flags,
  flag_type *__restrict__ dir,
  int *__restrict__ where,
  const BoxCU domain,
  int iter)
{
  int nx = domain.nx;
  int ny = domain.ny;
  int nz = domain.nz;

  for (int z = blockIdx.z; z < nz; z += gridDim.z) {
    for (int y = threadIdx.y + blockIdx.y*blockDim.y; y < ny; y += blockDim.y*gridDim.y) {
      for (int x = threadIdx.x + blockIdx.x*blockDim.x; x < nx; x += blockDim.x*gridDim.x) {
        if (flags[IDX(x, y, z, nx, ny, nz)] < bounce) {

          for (int i = 3; i < 3*nb_directions; i += 3){

            int xx = x + C_dirs[i    ];
            int yy = y + C_dirs[i + 1];
            int zz = z + C_dirs[i + 2];

            if (flags[IDX(xx, yy, zz, nx, ny, nz)] == moving_wall ) {
              flags[IDX(x, y, z, nx, ny, nz)] = wall_m;
              dir[IDX(x, y, z, nx, ny, nz)] = dir[IDX(xx, yy, zz, nx, ny, nz)];
              where[IDX(x, y, z, nx, ny, nz)]  |= (1 << IBAR((i/3), nb_directions));
            }
            if (flags[IDX(xx, yy, zz, nx, ny, nz)] == bounce ) {
              if (flags[IDX(x, y, z, nx, ny, nz)] != wall_m)flags[IDX(x, y, z, nx, ny, nz)] = wall;
              where[IDX(x, y, z, nx, ny, nz)]  |= (1 << IBAR((i/3), nb_directions));
            }
          }
        }
      }
    }
  }
}

__device__ __host__
inline void macroscopic(double *f, double* rho, u_type* u0, u_type* u1, u_type* u2)
{
  double X_M1 = f[ 1] + f[ 4] + f[ 5] + f[ 6] + f[ 7];
  double X_P1 = f[10] + f[13] + f[14] + f[15] + f[16];
  double X_0  = f[ 0] + f[ 2] + f[ 3] + f[ 8] + f[ 9] + f[11] + f[12] + f[17] + f[18];
  double Y_M1 = f[ 2] + f[ 4] + f[ 8] + f[ 9] + f[14];
  double Y_P1 = f[ 5] + f[11] + f[13] + f[17] + f[18];
  double Z_M1 = f[ 3] + f[ 6] + f[ 8] + f[16] + f[18];
  double Z_P1 = f[ 7] + f[ 9] + f[12] + f[15] + f[17];
  *rho = X_M1 + X_P1 + X_0;
  double one_over_rho = 1./ *rho;
  *u0 = (X_P1 - X_M1)* one_over_rho;
  *u1 = (Y_P1 - Y_M1)* one_over_rho;
  *u2 = (Z_P1 - Z_M1)* one_over_rho;
}

#define EQUILIBRIUM(rho, t, cu, usqr)  rho*(t)*(1 + (cu) + ((cu)*(cu))/2. - (usqr))
// Computes the second-order BGK equilibrium.
__device__
inline void d_equilibrium(double* fin, double rho, const float u0, const float u1, const float u2)
{
  double usqr = 3*(u0*u0 + u1*u1 + u2*u2)/2.;
  double cu;
  rho /= 36;
  fin[0 ] = EQUILIBRIUM(rho, 12, 0 , usqr);
  fin[1 ] = EQUILIBRIUM(rho, 2,3*( - u0          ), usqr);
  fin[10] = fin[1] - rho *12*(-u0);
  fin[2 ] = EQUILIBRIUM(rho, 2,3*(      - u1     ), usqr);
  fin[11] = fin[2] - rho*12*(-u1);
  fin[3 ] = EQUILIBRIUM(rho, 2,3*(           - u2), usqr);
  fin[12] = fin[3] - rho*12*(-u2);
  cu = 3 * (-u0 - u1);
  fin[4 ] = EQUILIBRIUM(rho, 1, cu, usqr);
  fin[13] = fin[4] - rho*2*cu;
  cu = 3 * (-u0 + u1);
  fin[5 ] = EQUILIBRIUM(rho, 1,cu, usqr);
  fin[14] = fin[5] -  rho*2*cu;
  cu = 3 * (-u0 - u2);
  fin[6 ] = EQUILIBRIUM(rho, 1,cu, usqr);
  fin[15] = fin[6] -  rho*2*cu;
  cu = 3*(-u0 + u2);
  fin[7 ] = EQUILIBRIUM(rho, 1,cu, usqr);
  fin[16] = fin[7] - rho*2*cu;
  cu = 3*(-u1 - u2);
  fin[8 ] = EQUILIBRIUM(rho, 1,cu, usqr);
  fin[17] = fin[8] - rho*2*cu;
  cu = 3*(-u1 + u2);
  fin[9 ] = EQUILIBRIUM(rho, 1, cu, usqr);
  fin[18] = fin[9] -  rho*2*cu;
}

template<lattice_type nb_directions>
__global__ void init_velocity_g(
  lbm_vars d_vars,
  BoxCU domain,
  BoxCU domain_vel,
  double depth,
  float u0 , float u1, float u2,
  double rho)
{
  int nx = domain.nx;
  int ny = domain.ny;
  int nz = domain.nz;

  double finl[nb_directions];

  for (int z = blockIdx.z; z < nz; z += gridDim.z) {
    for (int y = threadIdx.y + blockIdx.y*blockDim.y; y < ny; y += blockDim.y*gridDim.y) {
      for (int x = threadIdx.x + blockIdx.x*blockDim.x; x < nx; x += blockDim.x*gridDim.x) {

        int ugi = IDX(x - domain_vel.x0 + domain.x0,
            y - domain_vel.y0 + domain.y0,
            z - domain_vel.z0 + domain.z0,
            domain_vel.nx, domain_vel.ny, domain_vel.nz);

        d_vars.r[ugi] = rho;

        d_vars.u_star.u0[ugi] = u0;
        d_vars.u_star.u1[ugi] = u1;
        d_vars.u_star.u2[ugi] = u2;

        d_equilibrium(finl, rho, u0, u1, u2);

        for (int i = 0; i < nb_directions; ++i) {
          d_vars.f0[IDF(x, y, z, i, nx, ny, nz)] =  finl[i];
          d_vars.f1[IDF(x, y, z, i, nx, ny, nz)] =  finl[i];
        }
      }
    }
  }
}

__device__
inline void streaming(
  lbm_vars d_vars,
  double *fin, 
  const int x, const int y, const int z,
 const int nx, const int ny, const int nz)
{
  fin[0 ] = d_vars.f0[IDF(x   , y   , z   , 0 , nx, ny, nz)];
  fin[1 ] = d_vars.f0[IDF(x +1, y   , z   , 1 , nx, ny, nz)];
  fin[2 ] = d_vars.f0[IDF(x   , y +1, z   , 2 , nx, ny, nz)];
  fin[3 ] = d_vars.f0[IDF(x   , y   , z +1, 3 , nx, ny, nz)];
  fin[4 ] = d_vars.f0[IDF(x +1, y +1, z   , 4 , nx, ny, nz)];
  fin[5 ] = d_vars.f0[IDF(x +1, y -1, z   , 5 , nx, ny, nz)];
  fin[6 ] = d_vars.f0[IDF(x +1, y   , z +1, 6 , nx, ny, nz)];
  fin[7 ] = d_vars.f0[IDF(x +1, y   , z -1, 7 , nx, ny, nz)];
  fin[8 ] = d_vars.f0[IDF(x   , y +1, z +1, 8 , nx, ny, nz)];
  fin[9 ] = d_vars.f0[IDF(x   , y +1, z -1, 9 , nx, ny, nz)];
  fin[10] = d_vars.f0[IDF(x -1, y   , z   , 10, nx, ny, nz)];
  fin[11] = d_vars.f0[IDF(x   , y -1, z   , 11, nx, ny, nz)];
  fin[12] = d_vars.f0[IDF(x   , y   , z -1, 12, nx, ny, nz)];
  fin[13] = d_vars.f0[IDF(x -1, y -1, z   , 13, nx, ny, nz)];
  fin[14] = d_vars.f0[IDF(x -1, y +1, z   , 14, nx, ny, nz)];
  fin[15] = d_vars.f0[IDF(x -1, y   , z -1, 15, nx, ny, nz)];
  fin[16] = d_vars.f0[IDF(x -1, y   , z +1, 16, nx, ny, nz)];
  fin[17] = d_vars.f0[IDF(x   , y -1, z -1, 17, nx, ny, nz)];
  fin[18] = d_vars.f0[IDF(x   , y -1, z +1, 18, nx, ny, nz)];
}

__device__
inline void streaming_bounce(
  lbm_vars d_vars,
  double *fin,
  const int x, const int y, const int z,
  const int nx, const int ny, const int nz,
  int where)
{
  fin[0] = d_vars.f0[IDF(x, y, z, 0, nx, ny, nz)];
  if (where & (1 << 1)) {
    fin[1] = d_vars.f0[IDF(x, y, z, 10, nx, ny, nz)];
  }

  if (where & (1 << 2)) {
    fin[2] = d_vars.f0[IDF(x, y, z, 11, nx, ny, nz)];
  }

  if (where & (1 << 3)) {
    fin[3] = d_vars.f0[IDF(x, y, z, 12, nx, ny, nz)];
  }

  if (where & (1 << 4)) {
    fin[4] = d_vars.f0[IDF(x, y, z, 13, nx, ny, nz)];
  }

  if (where & (1 << 5)) {
    fin[5] = d_vars.f0[IDF(x, y, z, 14, nx, ny, nz)];
  }

  if (where & (1 << 6)) {
    fin[6] = d_vars.f0[IDF(x, y, z, 15, nx, ny, nz)];
  }

  if (where & (1 << 7)) {
    fin[7] = d_vars.f0[IDF(x, y, z, 16, nx, ny, nz)];
  }

  if (where & (1 << 8)) {
    fin[8] = d_vars.f0[IDF(x, y, z, 17, nx, ny, nz)];
  }

  if (where & (1 << 9)) {
    fin[9] = d_vars.f0[IDF(x, y, z, 18, nx, ny, nz)];
  }

  if (where & (1 << 10)) {
    fin[10] = d_vars.f0[IDF(x, y, z, 1, nx, ny, nz)];
  }

  if (where & (1 << 11)) {
    fin[11] = d_vars.f0[IDF(x, y, z, 2, nx, ny, nz)];
  }

  if (where & (1 << 12)) {
    fin[12] = d_vars.f0[IDF(x, y, z, 3, nx, ny, nz)];
  }

  if (where & (1 << 13)) {
    fin[13] = d_vars.f0[IDF(x, y, z, 4, nx, ny, nz)];
  }

  if (where & (1 << 14)) {
    fin[14] = d_vars.f0[IDF(x, y, z, 5, nx, ny, nz)];
  }

  if (where & (1 << 15)) {
    fin[15] = d_vars.f0[IDF(x, y, z, 6, nx, ny, nz)];
  }

  if (where & (1 << 16)) {
    fin[16] = d_vars.f0[IDF(x, y, z, 7, nx, ny, nz)];
  }

  if (where & (1 << 17)) {
    fin[17] = d_vars.f0[IDF(x, y, z, 8, nx, ny, nz)];
  }

  if (where & (1 << 18)) {
    fin[18] = d_vars.f0[IDF(x, y, z, 9, nx, ny, nz)];
  }
}

__device__ inline void streaming_wall2(
  lbm_vars d_vars,
  double *fin,
  const int x, const int y, const int z,
  const int nx, const int ny, const int nz,
  flag_type dir,
  flag_type* boundary,
  u_type u)
{
  u_type u0 = u;
  u_type u1 = 0.;
  u_type u2 = 0.;

  if  ( 0 || dir == 1){
    fin[1 ] +=  2*1./18*3*( - u0          );
  }
  if  ( 0 || dir == 3){
    fin[2 ] +=  2*1./18*3*(      - u1     );
  }
  if  ( 0 || dir == 5){
    fin[3 ] +=  2*1./18*3*(           - u2);
  }
  if  ( 0 || dir == 1|| dir == 3){
    fin[4 ] +=  2*1./36*3*( - u0 - u1     );
  }
  if  ( 0 || dir == 1|| dir == 2){
    fin[5 ] +=  2*1./36*3*( - u0 + u1     );
  }
  if  ( 0 || dir == 1|| dir == 5){
    fin[6 ] +=  2*1./36*3*( - u0      - u2);
  }
  if  ( 0 || dir == 1|| dir == 4){
    fin[7 ] +=  2*1./36*3*( - u0      + u2);
  }
  if  ( 0 || dir == 3|| dir == 5){
    fin[8 ] +=  2*1./36*3*(      - u1 - u2);
  }
  if  ( 0 || dir == 3|| dir == 4){
    fin[9 ] +=  2*1./36*3*(      - u1 + u2);
  }
  if  ( 0 || dir == 0){
    fin[10] +=  2*1./18*3*(   u0          );
  }
  if  ( 0 || dir == 2){
    fin[11] +=  2*1./18*3*(      + u1     );
  }
  if  ( 0 || dir == 4){
    fin[12] +=  2*1./18*3*(           + u2);
  }
  if  ( 0 || dir == 0|| dir == 2){
    fin[13] +=  2*1./36*3*(   u0 + u1     );
  }
  if  ( 0 || dir == 0|| dir == 3){
    fin[14] +=  2*1./36*3*(   u0 - u1     );
  }
  if  ( 0 || dir == 0|| dir == 4){
    fin[15] +=  2*1./36*3*(   u0      + u2);
  }
  if  ( 0 || dir == 0|| dir == 5){
    fin[16] +=  2*1./36*3*(   u0      - u2);
  }
  if  ( 0 || dir == 2|| dir == 4){
    fin[17] +=  2*1./36*3*(      + u1 + u2);
  }
  if  ( 0 || dir == 2|| dir == 5){
    fin[18] +=  2*1./36*3*(      + u1 - u2);
  }
}

template<lattice_type nb_directions>
__launch_bounds__(64)
__global__ void collide_and_stream_g(
  lbm_vars d_vars,
  const BoxCU domain,
  const double ulb,
  const double omega,
  bool out_u,
  int iter)
{
  int nx = domain.nx;
  int ny = domain.ny;
  int nz = domain.nz;

  for (int z = blockIdx.z; z < nz; z += gridDim.z) {
    for (int y = threadIdx.y + blockIdx.y*blockDim.y; y < ny; y += blockDim.y*gridDim.y) {
      for (int x = threadIdx.x + blockIdx.x*blockDim.x; x < nx; x += blockDim.x*gridDim.x) {

        double  finl[nb_directions], feq[nb_directions];
        u_type u0, u1, u2;
        double rho;

        flag_type boundary = d_vars.boundary_flag[IDX(x, y, z, nx, ny, nz)];

        if(boundary == bounce || boundary == moving_wall) continue;
        flag_type vel_dir  = d_vars.boundary_dirs[IDX(x, y, z, nx, ny, nz)];

        int ugi = IDX(x,y,z, domain.nx, domain.ny, domain.nz);

        streaming(d_vars, finl, x, y, z, nx, ny, nz);

        if (boundary == wall || boundary == wall_m){
          int where = d_vars.boundary_values[IDX(x, y, z, nx, ny, nz)];
          streaming_bounce(d_vars, finl, x, y, z, nx, ny, nz, where);
        }

        const u_type lid_vel = ulb;

        if (boundary == wall_m) {
          streaming_wall2(d_vars, finl, x, y, z, nx, ny, nz, vel_dir, d_vars.boundary_flag, lid_vel);
        }
        macroscopic(finl, &rho, &u0, &u1, &u2);

        if (boundary < bounce ) {

          d_equilibrium(feq, rho, u0, u1, u2);
          // BGK collision model.
          for (int i = 0; i < nb_directions; ++i){
            finl[i] = (1.-omega)*finl[i] +omega*feq[i];
          }
        }

        if(out_u && boundary < bounce){

          d_vars.u_star.u0[ugi] = u0;
          d_vars.u_star.u1[ugi] = u1;
          d_vars.u_star.u2[ugi] = u2;
        }

        for (int i = 0; i < nb_directions; ++i) {
          d_vars.f1[IDF(x, y, z, i, nx, ny, nz)] = finl[i];
        }
      }
    }
  }
}
