#define idx(i,j)   (i)*x_points+(j)

__global__ 
void core (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const double *__restrict__ u,
    const double *__restrict__ v,
    const int x_points,
    const int y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j < x_points - 1 && i < y_points - 1) {
    u_new[idx(i,j)] = u[idx(i,j)] + 
      (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
      (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
      (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
      (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

    v_new[idx(i,j)] = v[idx(i,j)] +
      (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
      (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
      (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
      (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
  }
}

__global__ 
void bound_h (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x_points) {
    u_new[idx(0,i)] = 1.0;
    v_new[idx(0,i)] = 1.0;
    u_new[idx(y_points-1,i)] = 1.0;
    v_new[idx(y_points-1,i)] = 1.0;
  }
}

__global__ 
void bound_v (
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < y_points) {
    u_new[idx(j,0)] = 1.0;
    v_new[idx(j,0)] = 1.0;
    u_new[idx(j,x_points-1)] = 1.0;
    v_new[idx(j,x_points-1)] = 1.0;
  }
}

__global__ 
void update (
    double *__restrict__ u,
    double *__restrict__ v,
    const double *__restrict__ u_new,
    const double *__restrict__ v_new,
    const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}

