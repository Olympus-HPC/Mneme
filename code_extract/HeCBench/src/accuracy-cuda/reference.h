int reference (
    const int N,
    const int D,
    const int top_k,
    const float* Xdata,
    const int* labelData)
{
  int count = 0;
  for (int row = 0; row < N; row++) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = 0; col < D; col++) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    if (ngt <= top_k) {
      ++count;
    }
  }
  return count;
}


