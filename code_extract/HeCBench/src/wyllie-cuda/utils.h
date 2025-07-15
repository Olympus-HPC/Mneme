#define NIL  -1
#define MASK 0xFFFFFFFF

// select an ordered list
void ordered_list (std::vector<int> &next) {
  int i;
  int nElems = next.size(); 
  for (i = 0; i < nElems-1; i++) {
    next[i] = i+1;
  }
  next[i] = NIL;
}

// select a random list
void random_list (std::vector<int> &next) {
  srand(123);
  int nElems = next.size(); 
  std::vector<bool> freeList (nElems);
  for (int i = 0; i < nElems; i++) freeList[i] = true;

  // randomly select the NIL position in between [1:nElems-1]
  int nil = rand() % (nElems-1) + 1;
  freeList[nil] = false;
  next[nil] = NIL;
  int prev_pos = nil;

  // randomly and uniquely select the next position in between [1:nElems-1]
  while (--nElems > 1) {
    int pos;
    do {
      pos = rand() % (next.size()-1) + 1;
    } while (freeList[pos] == false);
    freeList[pos] = false;
    next[pos] = prev_pos;
    prev_pos = pos;
  }
  next[0] = prev_pos; // the first element is always the head 
}

