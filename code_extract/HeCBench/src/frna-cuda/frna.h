#ifndef FRNA_H
#define FRNA_H

#include "fparam.h"
#include "int.h"

#ifdef __cplusplus
extern "C" {
#endif
struct frna
{
  int n; /* number of bases */
  fbase_t *seq;  /* sequence */
  int_t *v; /* n x n array */
  int_t *w, *wm, *wca; /* n x n arrays */
  int_t *w5, *w3; /* n elements */
};


typedef struct frna *frna_t;
frna_t frna_new(const char *s, fparam_t par);
void frna_delete(frna_t);
void frna_write(const frna_t, const char* fn);
  
#ifdef __cplusplus
}
#endif

#endif /* PRNA_H */
