#include "utils.h"

// "* 1" removes the warning:
// enumeral mismatch in conditional expression: ‘<anonymous enum>’ vs ‘<anonymous enum>’
__host__ __device__
inline int Dir_x( int octant ) { return octant & (1<<0) ? DIR_DN * 1: DIR_UP * 1; }
__host__ __device__
inline int Dir_y( int octant ) { return octant & (1<<1) ? DIR_DN * 1: DIR_UP * 1; }
__host__ __device__
inline int Dir_z( int octant ) { return octant & (1<<2) ? DIR_DN * 1 : DIR_UP * 1; }


int Arguments_exists( const Arguments*  args, const char* arg_name )
{
  int result = 0;
  int i = 0;

  for( i=0; i<args->argc; ++i )
  {
    if( args->argv_unconsumed[i] == NULL )
    {
      continue;
    }
    result = result || strcmp( args->argv_unconsumed[i], arg_name ) == 0;
  }

  return result;
}

/*===========================================================================*/
/* Process an argument of type int, remove from list---*/

int Arguments_consume_int_( Arguments*  args,
                            const char* arg_name )
{
  int result = 0;
  int found = 0;
  if( found ) {} /*---Remove unused var warning---*/
  int i = 0;

  for( i=0; i<args->argc; ++i )
  {
    if( args->argv_unconsumed[i] == NULL )
    {
      continue;
    }
    if( strcmp( args->argv_unconsumed[i], arg_name ) == 0 )
    {
      found = 1;
      args->argv_unconsumed[i] = NULL;
      ++i;
      assert( i<args->argc );
      result = atoi( args->argv_unconsumed[i] );
      args->argv_unconsumed[i] = NULL;
    }
  }
  return result;
}

/*===========================================================================*/
/* Consume an argument of type int, if not present then set to a default---*/

int Arguments_consume_int_or_default( Arguments*  args,
                                      const char* arg_name,
                                      int         default_value )
{
  assert( args != NULL );
  assert( arg_name != NULL );

  return Arguments_exists( args, arg_name ) ?
                     Arguments_consume_int_( args, arg_name ) : default_value;
}

/*===========================================================================*/
/* Pseudo-destructor for Arguments struct---*/
void Arguments_destroy( Arguments* args )
{
  assert( args != NULL );

  free( (void*) args->argv_unconsumed );
  if( args->argstring )
  {
    free( (void*) args->argstring );
  }
} /*---Arguments_destroy---*/

static inline int Quantities_scalefactor_space_( int ix_g, int iy_g, int iz_g )
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

static inline int Quantities_scalefactor_energy_( int ie, Dimensions dims )
{
  /*---Random power-of-two multiplier for each energy group,
       to help catch errors regarding indexing of energy groups.
  ---*/
  assert( ie >= 0 && ie < dims.ne );

  const int im = 714025;
  const int ia = 1366;
  const int ic = 150889;

  int result = ( (ie)*ia + ic ) % im;
  result = result & ( (1<<2) - 1 );
  result = 1 << result;

  return result;
}

static inline int Quantities_scalefactor_unknown_( int iu )
{
  /*---Random power-of-two multiplier for each cell unknown,
       to help catch errors regarding indexing of cell unknowns.
  ---*/
  assert( iu >= 0 && iu < NU );

  const int im = 312500;
  const int ia = 741;
  const int ic = 66037;

  int result = ( (iu)*ia + ic ) % im;
  result = result & ( (1<<2) - 1 );
  result = 1 << result;

  return result;
}

void initialize_input_state( P* const __restrict__   v,
                       const Dimensions        dims,
                       const int               nu)
{
  for( int iz=0; iz<dims.ncell_z; ++iz )
  for( int iy=0; iy<dims.ncell_y; ++iy )
  for( int ix=0; ix<dims.ncell_x; ++ix )
  for( int ie=0; ie<dims.ne; ++ie )
  for( int im=0; im<dims.nm; ++im )
  for( int iu=0; iu<nu; ++iu )
  {

    v[im + dims.nm      * (
      iu + nu           * (
      ix + dims.ncell_x * (
      iy + dims.ncell_y * (
      ie + dims.ne      * (
      iz + dims.ncell_z * ( /*---NOTE: This axis MUST be slowest-varying---*/
      0 ))))))] = 
             ( (P) (1 + im ) )
           * ( (P) Quantities_scalefactor_space_(ix, iy, iz) )
           * ( (P) Quantities_scalefactor_energy_( ie, dims ) )
           * ( (P) Quantities_scalefactor_unknown_( iu ) );
  }
}

size_t Dimensions_size_state( const Dimensions dims, int nu )
{
  return ( (size_t)dims.ncell_x )
       * ( (size_t)dims.ncell_y )
       * ( (size_t)dims.ncell_z )
       * ( (size_t)dims.ne )
       * ( (size_t)dims.nm )
       * ( (size_t)nu );
}

size_t Dimensions_size_facexy( const Dimensions dims, 
                               int nu, 
                               int num_face_octants_allocated )
{
  return ( (size_t)dims.ncell_x )
       * ( (size_t)dims.ncell_y )
       * ( (size_t)dims.ne )
       * ( (size_t)dims.na )
       * ( (size_t)nu )
       * ( (size_t)num_face_octants_allocated );
}

size_t Dimensions_size_facexz( const Dimensions dims, 
                               int nu, 
                               int num_face_octants_allocated )
{
  return ( (size_t)dims.ncell_x )
       * ( (size_t)dims.ncell_z )
       * ( (size_t)dims.ne )
       * ( (size_t)dims.na )
       * ( (size_t)nu )
       * ( (size_t)num_face_octants_allocated );
}

/*---------------------------------------------------------------------------*/

size_t Dimensions_size_faceyz( const Dimensions dims, 
                               int nu, 
                               int num_face_octants_allocated )
{
  return ( (size_t)dims.ncell_y )
       * ( (size_t)dims.ncell_z )
       * ( (size_t)dims.ne )
       * ( (size_t)dims.na )
       * ( (size_t)nu )
       * ( (size_t)num_face_octants_allocated );
}

int StepScheduler_nblock( const StepScheduler* stepscheduler )
{
  return stepscheduler->nblock_z_;
}

int StepScheduler_nstep( const StepScheduler* stepscheduler )
{
  int result = 0;  // no step on error

  switch( stepscheduler->nblock_octant_ )
  {
    case 8:
      result = 8 * StepScheduler_nblock( stepscheduler )
                                        + 2 * ( stepscheduler->nproc_x_ - 1 )
                                        + 3 * ( stepscheduler->nproc_y_ - 1 );
      break;

    case 4:
      result = 4 * StepScheduler_nblock( stepscheduler )
                                        + 1 * ( stepscheduler->nproc_x_ - 1 )
                                        + 2 * ( stepscheduler->nproc_y_ - 1 );
      break;

    case 2:
      result = 2 * StepScheduler_nblock( stepscheduler )
                                        + 1 * ( stepscheduler->nproc_x_ - 1 )
                                        + 1 * ( stepscheduler->nproc_y_ - 1 );
      break;

    case 1:
      result = 1 * StepScheduler_nblock( stepscheduler )
                                        + 1 * ( stepscheduler->nproc_x_ - 1 )
                                        + 1 * ( stepscheduler->nproc_y_ - 1 );
      break;

    default:
      printf("Error: unknown nblock octant %d. ", stepscheduler->nblock_octant_);
      printf("The value of next step is 0\n");
      break;
  }

  return result;
}

StepInfo StepScheduler_stepinfo( const StepScheduler* stepscheduler,  
                                 const int            step,
                                 const int            octant_in_block,
                                 const int            proc_x,
                                 const int            proc_y )
{
  assert( octant_in_block>=0 &&
          octant_in_block * stepscheduler->nblock_octant_ < NOCTANT );

  /*
  const int nblock_octant     = stepscheduler->nblock_octant_;
  */
  const int nproc_x           = stepscheduler->nproc_x_;
  const int nproc_y           = stepscheduler->nproc_y_;
  const int nblock            = StepScheduler_nblock( stepscheduler );
  const int nstep             = StepScheduler_nstep( stepscheduler );
  const int noctant_per_block = stepscheduler->noctant_per_block_;

  int octant_key    = 0;
  int wave          = 0;
  int step_base     = 0;
  int block         = 0;
  int octant        = 0;
  int dir_x         = 0;
  int dir_y         = 0;
  int dir_z         = 0;
  int start_x       = 0;
  int start_y       = 0;
  int start_z       = 0;
  int folded_octant = 0;
  int folded_block  = 0;

  StepInfo stepinfo;

  const int octant_selector[NOCTANT] = { 0, 4, 2, 6, 3, 7, 1, 5 };

  const int is_folded_x = noctant_per_block >= 2;
  const int is_folded_y = noctant_per_block >= 4;
  const int is_folded_z = noctant_per_block >= 8;

  const int folded_proc_x = ( is_folded_x && ( octant_in_block & (1<<0) ) )
                          ?  ( nproc_x - 1 - proc_x )
                          :                  proc_x;

  const int folded_proc_y = ( is_folded_y && ( octant_in_block & (1<<1) ) )
                          ?  ( nproc_y - 1 - proc_y )
                          :                  proc_y;

  /*===========================================================================
    For a given step and octant_in_block, the following computes the
    octant block (i.e., octant step), from which the octant can be
    computed, and the wavefront number, starting from the relevant begin
    corner of the selected octant.
    For the nblock_octant==8 case, the 8 octants are processed in sequence,
    in the order xyz = +++, ++-, -++, -+-, --+, ---, +-+, +--.
    This order is chosen to "pack" the wavefronts to minimize
    the KBA wavefront startup latency.
    For nblock_octant=k for some smaller k, this sequence is divided into
    subsequences of length k, and each subsequence defines the schedule
    for a given octant_in_block.
    The code below is essentially a search into the first subsequence
    to determine where the requested step is located.  Locations in
    the other subsequences can be derived from this.
    NOTE: the following does not address possibility that for a single
    step, two or more octants could update the same block.
  ===========================================================================*/

  wave = step - ( step_base );
  octant_key = 0;

  step_base += nblock;
  if ( step >= ( step_base + folded_proc_x
                           + folded_proc_y ) && ! is_folded_z )
  {
    wave = step - ( step_base );
    octant_key = 1;
  }
  step_base += nblock;
  if ( step >= ( step_base +            folded_proc_x
                           +            folded_proc_y ) && ! is_folded_y )
  {
    wave = step - ( step_base + (nproc_y-1) );
    octant_key = 2;
  }
  step_base += nblock + (nproc_y-1);
  if ( step >= ( step_base + (nproc_y-1-folded_proc_y)
                           +            folded_proc_x ) && ! is_folded_y )
  {
    wave = step - ( step_base );
    octant_key = 3;
  }
  step_base += nblock;
  if ( step >= ( step_base + (nproc_y-1-folded_proc_y)
                           +            folded_proc_x ) && ! is_folded_x )
  {
    wave = step - ( step_base + (nproc_x-1) );
    octant_key = 4;
  }
  step_base += nblock + (nproc_x-1);
  if ( step >= ( step_base + (nproc_y-1-folded_proc_y)
                           + (nproc_x-1-folded_proc_x) ) && ! is_folded_x )
  {
    wave = step - ( step_base );
    octant_key = 5;
  }
  step_base += nblock;
  if ( step >= ( step_base + (nproc_y-1-folded_proc_y)
                           + (nproc_x-1-folded_proc_x) ) && ! is_folded_x )
  {
    wave = step - ( step_base + (nproc_y-1) );
    octant_key = 6;
  }
  step_base += nblock + (nproc_y-1);
  if ( step >= ( step_base +            folded_proc_y
                           + (nproc_x-1-folded_proc_x) ) && ! is_folded_x )
  {
    wave = step - ( step_base );
    octant_key = 7;
  }

  folded_octant = octant_selector[ octant_key ];

  octant = folded_octant + octant_in_block;

  /*---Next convert the wavefront number to a block number based on
       location in the domain.  Use the equation that defines the plane.
  ---*/

  dir_x  = Dir_x( folded_octant );
  dir_y  = Dir_y( folded_octant );
  dir_z  = Dir_z( folded_octant );

  /*---Get coordinates of the starting corner block of the wavefront---*/
  start_x = dir_x==DIR_UP ? 0 : ( nproc_x - 1 );
  start_y = dir_y==DIR_UP ? 0 : ( nproc_y - 1 );
  start_z = dir_z==DIR_UP ? 0 : ( nblock  - 1 );

  /*---Get coordinate of block on this processor to be processed---*/
  folded_block = ( wave - ( start_x + folded_proc_x * dir_x )
                        - ( start_y + folded_proc_y * dir_y )
                        - ( start_z ) ) / dir_z;

  block = ( is_folded_z && ( octant_in_block & (1<<2) ) )
                          ? ( nblock - 1 - folded_block )
                          : folded_block;

  /*---Now determine whether the block calculation is active based on whether
       the block in question falls within the physical domain.
  ---*/

  stepinfo.is_active = block  >= 0 && block  < nblock &&
                       step   >= 0 && step   < nstep &&
                       proc_x >= 0 && proc_x < nproc_x &&
                       proc_y >= 0 && proc_y < nproc_y;

  /*---Set remaining values---*/

  stepinfo.block_z = stepinfo.is_active ? block : 0;
  stepinfo.octant  = octant;

  return stepinfo;
}

double get_time()
{
  struct timeval tv;
  int i = gettimeofday( &tv, NULL );
  double result = ( (double) tv.tv_sec * 1.e6 + (double) tv.tv_usec ); 
  return result;
}

double Quantities_flops_per_solve( const Dimensions dims )
{
  return 3. + 3. * NDIM;
}

/*---Size of state vector in angles space---*/

size_t Dimensions_size_state_angles( const Dimensions dims, int nu )
{
  return ( (size_t)dims.ncell_x )
       * ( (size_t)dims.ncell_y )
       * ( (size_t)dims.ncell_z )
       * ( (size_t)dims.ne )
       * ( (size_t)dims.na )
       * ( (size_t)nu )
       * ( (size_t)NOCTANT );
}
