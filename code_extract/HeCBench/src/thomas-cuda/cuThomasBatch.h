/**
 *
 *  @file cuThomasBatch.h
 *
 *  @brief cuThomasBatch kernel implementaion.
 *
 *  cuThomasBatch is a software package provided by
 *  Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *  @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *  @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

__global__ void cuThomasBatch(
            const double *L, const double *D, double *U, double *RHS,
            const int M,
            const int BATCHCOUNT);
