/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  basis.h                                                      
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

// allocate memory with failure checking
void* my_malloc(size_t bytes);

void* my_malloc_2(size_t bytes, std::string word);

// combination index
int ij2intindex(int i, int j);

// get nuclear charge of an element
int get_nuc_chg(char *element);

// get number of atoms 
int get_natoms(void);

// read geometry
void read_geom(Atom *p_atom);

// calculate nuclear repulsion energy
double calc_ene_nucl(Atom *p_atom);

// parse basis set; get number of basis functions
void parse_basis(Atom *p_atom, Basis *p_basis, int use_5d);

// read the full basis set created by parse_basis
void read_basis(Atom *p_atom, Basis *p_basis, int use_5d);

// print the basis set
void print_basis(Basis *p_basis);

// calculate one-electron integrals
double calc_int_overlap(Basis *p_basis, int a, int b);
double calc_int_kinetic(Basis *p_basis, int a, int b);
double calc_int_nuc_attr(Basis *p_basis, int a, int b, Atom *p_atom);

// calculate two-electron integrals
double calc_int_eri_rys(Basis *p_basis, int a, int b, int c, int d);
