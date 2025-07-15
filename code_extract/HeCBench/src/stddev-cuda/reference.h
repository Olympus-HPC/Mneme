template <typename Type, typename IdxType>
void stddev_ref(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  IdxType sample_size = sample ? N-1 : N;
  for (IdxType c = 0; c < D; c++) {
    Type sum = 0;
    for (IdxType r = 0; r < N; r++)
      sum += data[r*D+c] * data[r*D+c];
    std[c] = sqrtf(sum / sample_size);
  }
}
