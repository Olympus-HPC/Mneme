// https://www.geeksforgeeks.org/write-an-efficient-c-program-to-reverse-bits-of-a-number/
template <typename T>
T reverseBits(T num)
{
    unsigned int NO_OF_BITS = sizeof(T) * 8;
    T reverse_num = 0;
    unsigned int i;
    for (i = 0; i < NO_OF_BITS; i++) {
        if ((num & ((T)1 << i)))
            reverse_num |= (T)1 << ((NO_OF_BITS - 1) - i);
    }
    return reverse_num;
}

template <typename T>
T rev(T i, unsigned int nbits)
{
    if (sizeof(T) == 4 || nbits <= 32)
        return reverseBits(i) >> (8*sizeof(unsigned int) - nbits);
    else
        return reverseBits(i) >> (8*sizeof(unsigned long long) - nbits);
}

void bit_rev_cpu(fr_t* out, const fr_t *in, uint32_t lg_domain_size)
{
  uint32_t domain_size = 1 << lg_domain_size;
  #pragma omp parallel for
  for (uint32_t i = 0; i < domain_size; i++) {
    index_t r = rev(i, lg_domain_size);
    // assert(i == rev(r, lg_domain_size));
    if (i < r) {
      out[r] = in[i];
      out[i] = in[r];
    }
  }
}
