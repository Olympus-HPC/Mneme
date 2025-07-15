/******* dslash for improved KS fermions ****/

/* A stand-alone single-node validation of the GPU implementation */

#include "dslash.h"

//--------------------------------------------------------------------------------

static void make_temp_source_pointers(su3_vector **fwdsrc[4], su3_vector **bcksrc[4], 
                                      su3_vector **fwd3src[4], su3_vector **bck3src[4])
{
  for(int dir = 0; dir < 4; dir++){
    fwdsrc[dir] = (su3_vector **)malloc(sizeof(su3_vector *)*sites_on_node);
    bcksrc[dir] = (su3_vector **)malloc(sizeof(su3_vector *)*sites_on_node);
    fwd3src[dir] = (su3_vector **)malloc(sizeof(su3_vector *)*sites_on_node);
    bck3src[dir] = (su3_vector **)malloc(sizeof(su3_vector *)*sites_on_node);
    if (fwdsrc[dir] == NULL || bcksrc[dir] == NULL || fwd3src[dir] == NULL 
        || bck3src[dir] == NULL){
      std::cout << "Unable to allocate validation memory " << std::endl;
      exit(1);
    }
  }
}

//--------------------------------------------------------------------------------

static void destroy_temp_source_pointers(su3_vector **fwdsrc[4], su3_vector **bcksrc[4], 
                                         su3_vector **fwd3src[4], su3_vector **bck3src[4])
{
  for(int dir = 0; dir < 4; dir++){
    free(fwdsrc[dir]);
    free(bcksrc[dir]);
    free(fwd3src[dir]);
    free(bck3src[dir]);
  }
}

//--------------------------------------------------------------------------------

void dslash_fn_field(su3_vector *src, su3_vector *dst,
                     int parity, su3_matrix *fat, su3_matrix *lng,
                     su3_matrix *fatbck, su3_matrix *lngbck )
{
  su3_vector **fwdsrc[4], **bcksrc[4], **fwd3src[4], **bck3src[4];

  make_temp_source_pointers(fwdsrc, bcksrc, fwd3src, bck3src);

  // Set up gathers 

  for(int x = 0; x < nx; x++)
    for(int y = 0; y < ny; y++)
      for(int z = 0; z < nz; z++)
        for(int t = 0; t < nt; t++)
        {
          size_t i = node_index(x,y,z,t);
          fwdsrc[0][i]  = src + node_index(x+1,y,z,t);
          bcksrc[0][i]  = src + node_index(x-1,y,z,t);
          fwdsrc[1][i]  = src + node_index(x,y+1,z,t);
          bcksrc[1][i]  = src + node_index(x,y-1,z,t);
          fwdsrc[2][i]  = src + node_index(x,y,z+1,t);
          bcksrc[2][i]  = src + node_index(x,y,z-1,t);
          fwdsrc[3][i]  = src + node_index(x,y,z,t+1);
          bcksrc[3][i]  = src + node_index(x,y,z,t-1);
          fwd3src[0][i] = src + node_index(x+3,y,z,t);
          bck3src[0][i] = src + node_index(x-3,y,z,t);
          fwd3src[1][i] = src + node_index(x,y+3,z,t);
          bck3src[1][i] = src + node_index(x,y-3,z,t);
          fwd3src[2][i] = src + node_index(x,y,z+3,t);
          bck3src[2][i] = src + node_index(x,y,z-3,t);
          fwd3src[3][i] = src + node_index(x,y,z,t+3);
          bck3src[3][i] = src + node_index(x,y,z,t-3);
        }

  // Even parity only for now

#pragma omp parallel for
  for(size_t i = 0; i < even_sites_on_node; i++){
    su3_vector tvec;
    mult_su3_mat_vec_sum_4dir( fat + 4*i,
                               fwdsrc[XUP][i], fwdsrc[YUP][i], 
                               fwdsrc[ZUP][i], fwdsrc[TUP][i], 
                               dst + i);
    mult_su3_mat_vec_sum_4dir( lng + 4*i,
                               fwd3src[XUP][i], fwd3src[YUP][i], 
                               fwd3src[ZUP][i], fwd3src[TUP][i], 
                               &tvec );
    add_su3_vector(dst + i, &tvec, dst + i);
  }

#pragma omp parallel for
  for(size_t i = 0; i < even_sites_on_node; i++){
    su3_vector tvec;

    mult_su3_mat_vec_sum_4dir( fatbck + 4*i,
                               bcksrc[XUP][i], bcksrc[YUP][i], 
                               bcksrc[ZUP][i], bcksrc[TUP][i], 
                               &tvec);
    sub_su3_vector(dst + i, &tvec, dst + i);
    mult_su3_mat_vec_sum_4dir( lngbck + 4*i,
                               bck3src[XUP][i], bck3src[YUP][i], 
                               bck3src[ZUP][i], bck3src[TUP][i], 
                               &tvec);
    sub_su3_vector(dst + i, &tvec, dst + i );
  }

  destroy_temp_source_pointers(fwdsrc, bcksrc, fwd3src, bck3src);
}
