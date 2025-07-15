void ccsd_tengy_gpu(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                    const double * __restrict__ f2n,    const double * __restrict__ f2t,
                    const double * __restrict__ f3n,    const double * __restrict__ f3t,
                    const double * __restrict__ f4n,    const double * __restrict__ f4t,
                    const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                    const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                    const double * __restrict__ eorb,   const double eaijk,
                    double * __restrict__ emp4i, double * __restrict__ emp5i,
                    double * __restrict__ emp4k, double * __restrict__ emp5k,
                    const int ncor, const int nocc, const int nvir);

void ccsd_trpdrv(double * __restrict__ f1n, double * __restrict__ f1t,
                 double * __restrict__ f2n, double * __restrict__ f2t,
                 double * __restrict__ f3n, double * __restrict__ f3t,
                 double * __restrict__ f4n, double * __restrict__ f4t,
                 double * __restrict__ eorb,
                 int    * __restrict__ ncor_, int * __restrict__ nocc_, int * __restrict__ nvir_,
                 double * __restrict__ emp4_, double * __restrict__ emp5_,
                 int    * __restrict__ a_, int * __restrict__ i_, int * __restrict__ j_, int * __restrict__ k_, int * __restrict__ klo_,
                 double * __restrict__ tij, double * __restrict__ tkj, double * __restrict__ tia, double * __restrict__ tka,
                 double * __restrict__ xia, double * __restrict__ xka, double * __restrict__ jia, double * __restrict__ jka,
                 double * __restrict__ kia, double * __restrict__ kka, double * __restrict__ jij, double * __restrict__ jkj,
                 double * __restrict__ kij, double * __restrict__ kkj,
                 double * __restrict__ dintc1, double * __restrict__ dintx1, double * __restrict__ t1v1,
                 double * __restrict__ dintc2, double * __restrict__ dintx2, double * __restrict__ t1v2)
{
    double emp4 = *emp4_;
    double emp5 = *emp5_;

    double emp4i = 0.0;
    double emp5i = 0.0;
    double emp4k = 0.0;
    double emp5k = 0.0;

    const int ncor = *ncor_;
    const int nocc = *nocc_;
    const int nvir = *nvir_;

    /* convert from Fortran to C offset convention... */
    const int k   = *k_ - 1;
    //const int klo = *klo_ - 1;
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;

    const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    ccsd_tengy_gpu(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
                   dintc1, dintx1, t1v1, dintc2, dintx2, t1v2,
                   eorb, eaijk, &emp4i, &emp5i, &emp4k, &emp5k,
                   ncor, nocc, nvir);

    emp4 += emp4i;
    emp5 += emp5i;

    if (*i_ != *k_) {
        emp4 += emp4k;
        emp5 += emp5k;
    }

    *emp4_ = emp4;
    *emp5_ = emp5;

    return;
}
