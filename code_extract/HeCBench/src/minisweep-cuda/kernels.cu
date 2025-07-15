__device__
int Quantities_scalefactor_space_acceldir(int ix_g, int iy_g, int iz_g)
{
  int result = 0;

#ifndef RELAXED_TESTING
  const int im = 134456;
  const int ia = 8121;
  const int ic = 28411;

  result = ( (result+(ix_g+2))*ia + ic ) % im;
  result = ( (result+(iy_g+2))*ia + ic ) % im;
  result = ( (result+(iz_g+2))*ia + ic ) % im;
  result = ( (result+(ix_g+3*iy_g+7*iz_g+2))*ia + ic ) % im;
  result = ix_g+3*iy_g+7*iz_g+2;
  result = result & ( (1<<2) - 1 );
#endif
  result = 1 << result;

  return result;
}

__device__
P Quantities_init_face_acceldir(int ia, int ie, int iu, int scalefactor_space, int octant)
{
  /*--- Quantities_affinefunction_ inline ---*/
  return ( (P) (1 + ia) ) 

    /*--- Quantities_scalefactor_angle_ inline ---*/
    * ( (P) (1 << (ia & ( (1<<3) - 1))) ) 

    /*--- Quantities_scalefactor_space_ inline ---*/
    * ( (P) scalefactor_space)

    /*--- Quantities_scalefactor_energy_ inline ---*/
    * ( (P) (1 << ((( (ie) * 1366 + 150889) % 714025) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_unknown_ inline ---*/
    * ( (P) (1 << ((( (iu) * 741 + 60037) % 312500) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_octant_ ---*/
    * ( (P) 1 + octant);
}

__device__
void Quantities_solve_acceldir(P* vs_local, Dimensions dims, P* facexy, P* facexz, P* faceyz,
                             int ix, int iy, int iz,
                             int ix_g, int iy_g, int iz_g,
                             int ie, int ia,
                             int octant, int octant_in_block, int noctant_per_block)
{
  const int dir_x = Dir_x( octant );
  const int dir_y = Dir_y( octant );
  const int dir_z = Dir_z( octant );

  int iu = 0;

  /*---Average the face values and accumulate---*/

  /*---The state value and incoming face values are first adjusted to
    normalized values by removing the spatial scaling.
    They are then combined using a weighted average chosen in a special
    way to give just the expected result.
    Finally, spatial scaling is applied to the result which is then
    stored.
    ---*/

  /*--- Quantities_scalefactor_octant_ inline ---*/
  const P scalefactor_octant = 1 + octant;
  const P scalefactor_octant_r = ((P)1) / scalefactor_octant;

  /*---Quantities_scalefactor_space_ inline ---*/
  const P scalefactor_space = (P)Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);
  const P scalefactor_space_r = ((P)1) / scalefactor_space;
  const P scalefactor_space_x_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g - dir_x, iy_g, iz_g );
  const P scalefactor_space_y_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g, iy_g - dir_y, iz_g );
  const P scalefactor_space_z_r = ((P)1) /
    Quantities_scalefactor_space_acceldir( ix_g, iy_g, iz_g - dir_z );

#ifdef USE_OPENMP_TARGET
// no equivalent
#elif defined(USE_ACC)
#pragma acc loop seq
#endif
  for( iu=0; iu<NU; ++iu )
    {

      int vs_local_index = ia + dims.na * (
                           iu + NU  * (
                           ie + dims.ne * (
                           ix + dims.ncell_x * (
                           iy + dims.ncell_y * (
                           octant + NOCTANT * (
                           0))))));

      const P result = ( vs_local[vs_local_index] * scalefactor_space_r + 
               (
                /*--- ref_facexy inline ---*/
                facexy[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        ix + dims.ncell_x * (
                        iy + dims.ncell_y * (
                        octant + NOCTANT * (
                        0 )))))) ]

               /*--- Quantities_xfluxweight_ inline ---*/
               * (P) ( 1 / (P) 2 )

               * scalefactor_space_z_r

               /*--- ref_facexz inline ---*/
               + facexz[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        ix + dims.ncell_x * (
                        iz + dims.ncell_z * (
                        octant + NOCTANT * (
                        0 )))))) ]

               /*--- Quantities_yfluxweight_ inline ---*/
               * (P) ( 1 / (P) 4 )

               * scalefactor_space_y_r

               /*--- ref_faceyz inline ---*/
               + faceyz[ia + dims.na      * (
                        iu + NU           * (
                        ie + dims.ne      * (
                        iy + dims.ncell_y * (
                        iz + dims.ncell_z * (
                        octant + NOCTANT * (
                        0 )))))) ]

                        /*--- Quantities_zfluxweight_ inline ---*/
                        * (P) ( 1 / (P) 4 - 1 / (P) (1 << ( ia & ( (1<<3) - 1 ) )) )

               * scalefactor_space_x_r
               ) 
               * scalefactor_octant_r ) * scalefactor_space;

      vs_local[vs_local_index] = result;

      const P result_scaled = result * scalefactor_octant;
      /*--- ref_facexy inline ---*/
      facexy[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             ix + dims.ncell_x * (
             iy + dims.ncell_y * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

      /*--- ref_facexz inline ---*/
      facexz[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             ix + dims.ncell_x * (
             iz + dims.ncell_z * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

      /*--- ref_faceyz inline ---*/
      faceyz[ia + dims.na      * (
             iu + NU           * (
             ie + dims.ne      * (
             iy + dims.ncell_y * (
             iz + dims.ncell_z * (
             octant + NOCTANT * (
             0 )))))) ] = result_scaled;

    } /*---for---*/
}

__device__
void Sweeper_sweep_cell_acceldir( const Dimensions &dims,
                                  int wavefront,
                                  int octant,
                                  int ix, int iy,
                                  int ix_g, int iy_g, int iz_g,
                                  int dir_x, int dir_y, int dir_z,
                                  P* __restrict__ facexy,
                                  P* __restrict__ facexz,
                                  P* __restrict__ faceyz,
                                  const P* __restrict__ a_from_m,
                                  const P* __restrict__ m_from_a,
                                  const P* __restrict__ vi,
                                  P* __restrict__ vo,
                                  P* __restrict__ vs_local,
                                  int octant_in_block,
                                  int noctant_per_block,
                                  int ie)
{
  /*---Declarations---*/
//  int iz = 0;
//  int ie = 0;
  int im = 0;
  int ia = 0;
  int iu = 0;
  /* int octant = 0; */

  /*--- Dimensions ---*/
  int dims_ncell_x = dims.ncell_x;
  int dims_ncell_y = dims.ncell_y;
  int dims_ncell_z = dims.ncell_z;
  int dims_ne = dims.ne;
  int dims_na = dims.na;
  int dims_nm = dims.nm;

  /*--- Solve for Z dimension, and check bounds.
    The sum of the dimensions should equal the wavefront number.
    If z < 0 or z > wavefront number, we are out of bounds.
    Z also shouldn't exceed the spacial bound for the z dimension.

    The calculation is adjusted for the direction of each axis
    in a given octant.
  ---*/

  const int ixwav = dir_x==DIR_UP ? ix : (dims_ncell_x-1) - ix;
  const int iywav = dir_y==DIR_UP ? iy : (dims_ncell_y-1) - iy;
  const int izwav = wavefront - ixwav - iywav;
  const int iz = dir_z==DIR_UP ? izwav : (dims_ncell_z-1) - izwav;

//  int ixwav, iywav, izwav;
//  if (dir_x==DIR_UP) { ixwav = ix; } else { ixwav = (dims_ncell_x-1) - ix; }
//  if (dir_y==DIR_UP) { iywav = iy; } else { iywav = (dims_ncell_y-1) - iy; }
  
//  if (dir_z==DIR_UP) {
//    iz = wavefront - (ixwav + iywav); } 
//  else { 
//    iz = (dims_ncell_z-1) - (wavefront - (ixwav + iywav));
//  }

  /*--- Bounds check ---*/
  if ((iz >= 0 && iz < dims_ncell_z) )// &&
    /* ((dir_z==DIR_UP && iz <= wavefront) || */
    /*  (dir_z==DIR_DN && (dims_ncell_z-1-iz) <= wavefront))) */
    {

   /*---Loop over energy groups---*/
//      for( ie=0; ie<dims_ne; ++ie )
      {

      /*--------------------*/
      /*---Transform state vector from moments to angles---*/
      /*--------------------*/

      /*---This loads values from the input state vector,
           does the small dense matrix-vector product,
           and stores the result in a relatively small local
           array that is hopefully small enough to fit into
           processor cache.
      ---*/

      for( iu=0; iu<NU; ++iu )
      for( ia=0; ia<dims_na; ++ia )
      { 
        // reset reduction
        P result = (P)0;
        for( im=0; im < dims_nm; ++im )
        {
          /*--- const_ref_a_from_m inline ---*/
          result += a_from_m[ ia     + dims_na * (
                              im     +      NM * (
                              octant + NOCTANT * (
                              0 ))) ] * 

            /*--- const_ref_state inline ---*/
            vi[im + dims.nm      * (
                          iu + NU           * (
                          ix + dims_ncell_x * (
                          iy + dims_ncell_y * (
                          ie + dims_ne      * (
                          iz + dims_ncell_z * ( /*---NOTE: This axis MUST be slowest-varying---*/
                          0 ))))))];
        }

        /*--- ref_vslocal inline ---*/
        vs_local[ ia + dims.na * (
                  iu + NU  * (
                  ie + dims_ne * (
                  ix + dims_ncell_x * (
                  iy + dims_ncell_y * (
                  octant + NOCTANT * (
                                       0)))))) ] = result;
      }
      }

      /*--------------------*/
      /*---Perform solve---*/
      /*--------------------*/

//   /*---Loop over energy groups---*/
      for( ia=0; ia<dims_na; ++ia )
      {
        Quantities_solve_acceldir(vs_local, dims, facexy, facexz, faceyz, 
                             ix, iy, iz,
                             ix_g, iy_g, iz_g,
                             ie, ia,
                             octant, octant_in_block, noctant_per_block);
      }

      /*--------------------*/
      /*---Transform state vector from angles to moments---*/
      /*--------------------*/

      /*---Perform small dense matrix-vector products and store
           the result in the output state vector.
      ---*/

   /*---Loop over energy groups---*/
      for( iu=0; iu<NU; ++iu )
      for( im=0; im<dims_nm; ++im )
      {
        P result = (P)0;
        for( ia=0; ia<dims_na; ++ia )
        {
         /*--- const_ref_m_from_a ---*/
         result += m_from_a[ im     +      NM * (
                             ia     + dims_na * (
                             octant + NOCTANT * (
                             0 ))) ] *

         /*--- const_ref_vslocal ---*/
         vs_local[ ia + dims_na * (
                   iu + NU    * (
                   ie + dims_ne * (
                   ix + dims_ncell_x * (
                   iy + dims_ncell_y * (
                   octant + NOCTANT * (
                   0 )))))) ];
        }

        /*--- ref_state inline ---*/
        atomicAdd(
        &vo[im + dims.nm     * (
           iu + NU           * (
           ix + dims_ncell_x * (
           iy + dims_ncell_y * (
           ie + dims_ne      * (
           iz + dims_ncell_z * ( /*---NOTE: This axis MUST be slowest-varying---*/
           0 ))))))] , result);
      }

//      } /*---ie---*/

    } /*--- iz ---*/
}
__global__ void init_facexy(
    const int ix_base, 
    const int iy_base,
    const int dims_b_ne,
    const int dims_b_na,
    const int dims_b_ncell_x,
    const int dims_b_ncell_y,
    const int dims_b_ncell_z,
    const int dims_ncell_z,
    P* facexy) 
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x; 
  int iy = blockDim.y * blockIdx.y + threadIdx.y; 
  int octant = blockDim.z * blockIdx.z + threadIdx.z; 
  if (ix >= dims_b_ncell_x || iy >= dims_b_ncell_y || octant >= NOCTANT) return;

  for(int ie=0; ie<dims_b_ne; ++ie )
    for(int iu=0; iu<NU; ++iu )
      for(int ia=0; ia<dims_b_na; ++ia )
      {
        const int dir_z = Dir_z( octant );
        const int iz = dir_z == DIR_UP ? -1 : dims_b_ncell_z;

        const int ix_g = ix + ix_base; // dims_b_ncell_x * proc_x;
        const int iy_g = iy + iy_base; // dims_b_ncell_y * proc_y;
        const int iz_g = iz + (dir_z == DIR_UP ? 0 : dims_ncell_z - dims_b_ncell_z);
        //const int iz_g = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;

        /*--- Quantities_scalefactor_space_ inline ---*/
        const int scalefactor_space
          = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);

        /*--- ref_facexy inline ---*/
        facexy[FACEXY_ADDR(dims_b_ncell_x, dims_b_ncell_y)]
          /*--- Quantities_init_face routine ---*/
          = Quantities_init_face_acceldir(ia, ie, iu, scalefactor_space, octant);
        //printf("kernel facexy: %d %d %d %d %d %f\n", 
          //ia, ie, iu, scalefactor_space, octant,
          //Quantities_init_face_acceldir(ia, ie, iu, scalefactor_space, octant));
      } /*---for---*/
}

__global__ void init_facexz(
    const int ix_base, 
    const int iy_base,
    const int dims_b_ne,
    const int dims_b_na,
    const int dims_b_ncell_x,
    const int dims_b_ncell_y,
    const int dims_b_ncell_z,
    const int proc_y_min,
    const int proc_y_max,
    StepInfoAll stepinfoall,
    P* facexz) 
{
  int ix = blockDim.x * blockIdx.x + threadIdx.x; 
  int iz = blockDim.y * blockIdx.y + threadIdx.y; 
  int octant = blockDim.z * blockIdx.z + threadIdx.z; 
  if (ix >= dims_b_ncell_x || iz >= dims_b_ncell_z || octant >= NOCTANT) return;

  for(int ie=0; ie<dims_b_ne; ++ie )
    for(int iu=0; iu<NU; ++iu )
      for(int ia=0; ia<dims_b_na; ++ia )
      {
        const int dir_y = Dir_y( octant );
        const int iy = dir_y == DIR_UP ? -1 : dims_b_ncell_y;

        const int ix_g = ix + ix_base; // dims_b_ncell_x * proc_x;
        const int iy_g = iy + iy_base; // dims_b_ncell_y * proc_y;
        const int iz_g = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;

        if ((dir_y == DIR_UP && proc_y_min) || (dir_y == DIR_DN && proc_y_max)) {

          /*--- Quantities_scalefactor_space_ inline ---*/
          const int scalefactor_space
            = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);

          /*--- ref_facexz inline ---*/
        facexz[FACEXZ_ADDR(dims_b_ncell_x, dims_b_ncell_z)]
            /*--- Quantities_init_face routine ---*/
            = Quantities_init_face_acceldir(ia, ie, iu, scalefactor_space, octant);
        } /*---if---*/
      } /*---for---*/
}

__global__ void init_faceyz(
    const int ix_base, 
    const int iy_base,
    const int dims_b_ne,
    const int dims_b_na,
    const int dims_b_ncell_x,
    const int dims_b_ncell_y,
    const int dims_b_ncell_z,
    const int proc_x_min,
    const int proc_x_max,
    StepInfoAll stepinfoall,
    P* faceyz)
{
  int iy = blockDim.x * blockIdx.x + threadIdx.x; 
  int iz = blockDim.y * blockIdx.y + threadIdx.y; 
  int octant = blockDim.z * blockIdx.z + threadIdx.z; 
  if (iy >= dims_b_ncell_y || iz >= dims_b_ncell_z || octant >= NOCTANT) return;

  for(int ie=0; ie<dims_b_ne; ++ie )
    for(int iu=0; iu<NU; ++iu )
      for(int ia=0; ia<dims_b_na; ++ia )
      {

        const int dir_x = Dir_x( octant );
        const int ix = dir_x == DIR_UP ? -1 : dims_b_ncell_x;

        const int ix_g = ix + ix_base; // dims_b_ncell_x * proc_x;
        const int iy_g = iy + iy_base; // dims_b_ncell_y * proc_y;
        const int iz_g = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;

        if ((dir_x == DIR_UP && proc_x_min) || (dir_x == DIR_DN && proc_x_max)) {

          /*--- Quantities_scalefactor_space_ inline ---*/
          const int scalefactor_space
            = Quantities_scalefactor_space_acceldir(ix_g, iy_g, iz_g);

          /*--- ref_faceyz inline ---*/
          faceyz[FACEYZ_ADDR(dims_b_ncell_y, dims_b_ncell_z)]
            /*--- Quantities_init_face routine ---*/
            = Quantities_init_face_acceldir(ia, ie, iu, scalefactor_space, octant);
        } /*---if---*/
      } /*---for---*/
}

__global__ void wavefronts(
    const int num_wavefronts,  
    const int ix_base, 
    const int iy_base,
    const int v_b_size,
    const int noctant_per_block,
    const Dimensions dims_b,
    StepInfoAll stepinfoall,
    P*__restrict__ facexy, 
    P*__restrict__ facexz, 
    P*__restrict__ faceyz, 
    P*__restrict__ a_from_m,
    P*__restrict__ m_from_a,
    P*__restrict__ vi,
    P*__restrict__ vo,
    P*__restrict__ vs_local)
{
  int octant = blockDim.x * blockIdx.x + threadIdx.x; 
  int ie = blockDim.y * blockIdx.y + threadIdx.y; 
  if (ie >= dims_b.ne || octant >= NOCTANT) return;

  const int dims_b_ncell_x = dims_b.ncell_x;
  const int dims_b_ncell_y = dims_b.ncell_y;
  const int dims_b_ncell_z = dims_b.ncell_z;

  /*--- Loop over wavefronts ---*/
  for (int wavefront = 0; wavefront < num_wavefronts; wavefront++)
  {
    for( int iywav=0; iywav<dims_b_ncell_y; ++iywav )
      for( int ixwav=0; ixwav<dims_b_ncell_x; ++ixwav )
      {

        if (stepinfoall.stepinfo[octant].is_active) {

          /*---Decode octant directions from octant number---*/

          const int dir_x = Dir_x( octant );
          const int dir_y = Dir_y( octant );
          const int dir_z = Dir_z( octant );

          const int octant_in_block = octant;

          const int ix = dir_x==DIR_UP ? ixwav : dims_b_ncell_x - 1 - ixwav;
          const int iy = dir_y==DIR_UP ? iywav : dims_b_ncell_y - 1 - iywav;
          const int izwav = wavefront - ixwav - iywav;
          const int iz = dir_z==DIR_UP ? izwav : (dims_b_ncell_z-1) - izwav;

          const int ix_g = ix + ix_base; // dims_b_ncell_x * proc_x;
          const int iy_g = iy + iy_base; // dims_b_ncell_y * proc_y;
          const int iz_g = iz + stepinfoall.stepinfo[octant].block_z * dims_b_ncell_z;

          const int v_offset = stepinfoall.stepinfo[octant].block_z * v_b_size;

          /*--- In-gridcell computations ---*/
          Sweeper_sweep_cell_acceldir( dims_b, wavefront, octant, ix, iy,
              ix_g, iy_g, iz_g,
              dir_x, dir_y, dir_z,
              facexy, facexz, faceyz,
              a_from_m, m_from_a,
              &(vi[v_offset]), &(vo[v_offset]), vs_local,
              octant_in_block, noctant_per_block, ie );

        } /*---if---*/

      } /*---octant/ix/iy---*/

  } /*--- wavefront ---*/
} 
