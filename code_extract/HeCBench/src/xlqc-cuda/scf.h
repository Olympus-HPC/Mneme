/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  scf.h                                                      
 License:   BSD 3-Clause License

 This software is provided by the copyright holders and contributors "as is"
 and any express or implied warranties, including, but not limited to, the
 implied warranties of merchantability and fitness for a particular purpose are
 disclaimed. In no event shall the copyright holder or contributors be liable
 for any direct, indirect, incidental, special, exemplary, or consequential
 damages (including, but not limited to, procurement of substitute goods or
 services; loss of use, data, or profits; or business interruption) however
 caused and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of the use
 of this software, even if advised of the possibility of such damage.
 *****************************************************************************/

// matrix inner product
double mat_inn_prod(int DIM, double **A, double **B);

// GSL eigen solver for real symmetric matrix
void my_eigen_symmv(gsl_matrix* data, int DIM,
                    gsl_vector* eval, gsl_matrix* evec);

// print gsl matrix
void my_print_matrix(gsl_matrix* A);

// print gsl vector
void my_print_vector(gsl_vector* x);

// get core Hamiltonian
void sum_H_core(int nbasis, gsl_matrix *H_core, gsl_matrix *T, gsl_matrix *V);

// diagonalize overlap matrix
void diag_overlap(int nbasis, gsl_matrix *S, gsl_matrix *S_invsqrt);

// from Fock matrix to MO coeffcients
void Fock_to_Coef(int nbasis, gsl_matrix *Fock, gsl_matrix *S_invsqrt, 
                  gsl_matrix *Coef, gsl_vector *emo);

// from MO coeffcients to density matrix
void Coef_to_Dens(int nbasis, int n_occ, gsl_matrix *Coef, gsl_matrix *D);

// compute the initial SCF energy
double get_elec_ene(int nbasis, gsl_matrix *D, gsl_matrix *H_core, 
                    gsl_matrix *Fock);

// form Fock matrix
void form_Fock(int nbasis, gsl_matrix *H_core, gsl_matrix *J, gsl_matrix *K, gsl_matrix *Fock);

// Generalized Wolfsberg-Helmholtz initial guess
void init_guess_GWH(Basis *p_basis, gsl_matrix *H_core, gsl_matrix *S, gsl_matrix *Fock);

// DIIS
void update_Fock_DIIS(int *p_diis_dim, int *p_diis_index, double *p_delta_DIIS, 
                      gsl_matrix *Fock, gsl_matrix *D_prev, gsl_matrix *S, Basis *p_basis,
                      double ***diis_err, double ***diis_Fock);
