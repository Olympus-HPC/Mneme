#include "utils.h"

/* compare two floats */
int floatcomp(const void* elem1, const void* elem2){
  if(*(const float*)elem1 < *(const float*)elem2)
    return -1;
  return *(const float*)elem1 > *(const float*)elem2;
}

void fgoleft(findex_t *frag, int ar){
  frag->i -= 1;
  if(frag->i < 0){
    frag->f -= 1;
    frag->i = ar - 1;
  }
  if(frag->f < 0){
    *frag = (findex_t){-1,-1};
  }
}

/* get the left frag position */
findex_t fgetleft(findex_t frag, int ar){
  findex_t out = frag;
  out.i -= 1;
  if(out.i < 0){
    out.f -= 1;
    out.i = ar - 1;
  }
  if(out.f < 0){
    out = (findex_t){-1,-1};
  }
  return out;
}

/* put the new value well distributed in a GPU */
void newtemp(float *aT, int *ar, int *R, findex_t l){
  findex_t left = fgetleft(l, *ar);
  float ntemp = (aT[l.i] + aT[left.i])/2.0f;
  //printf("new Temp = (%f + %f)/2 = %f\n", s->aT[l.f][l.i], s->aT[left.f][left.i], ntemp);
  aT[*ar] = ntemp;
  (*ar)++;
  /* update the number of active replicas */
  (*R)++;
  //printf("new R = %i\n", s->R);
}

/* rebuild the temperatures, sorted */
void rebuild_temps(float *aT, int R, int ar){
  int count  = 0;
  float *flat = (float*)malloc(sizeof(float)*R);
  /* flatten the temperatures */
  for(int j=0; j<ar; ++j){
    flat[count++] = aT[j];
  }

  /* sort them */
  qsort(flat, R, sizeof(float), floatcomp);

  /* fragment the sorted temperatures */
  count = 0;
  for(int j=0; j<ar; ++j){
    aT[j] = flat[count++];
  }
  free(flat);
}

/* insert temperatures at the "ins" lowest exchange places */
void insert_temps(float *aavex, float *aT, int *R, int *ar, int ains){
  /* minheap */
  minHeap hp = initMinHeap(0);
  /* put average exchange rates in a min heap */
  for(int j = *ar-1; j > 0; --j){
    //printf("inserting [%f, {%i ,%i}] \n", s->aavex[i][j], i, j);
    insertNode(&hp, aavex[j], (findex_t){0, j});
  }
  //printf("heap has size = %i      s->ains = %i\n", hp.size, s->ains);
  /* get the lowest "ins" exchange rates */
  for(int i = 0; i < ains && hp.size > 0; ++i){
    node nod = popRoot(&hp);
    //printf("heap now is size  = %i:\n", hp.size);
    //levelorderTraversal(&hp);
    newtemp(aT, ar, R, nod.coord);
    //printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "aT");
  }
}

/* rebuild atrs and arts indices */
void rebuild_indices(findex_t* arts, findex_t *atrs, int ar) {
  for(int j = 0; j < ar; ++j){
    arts[j] = atrs[j] = (findex_t){0, j};
  }
}


void printarray(float *a, int n, const char *name){
  std::cout << name << "\t = [";
  for(int i = 0; i < n; ++i){
    std::cout << a[i] << ", ";
  }
  printf("]\n");
}

/* print indexed fragmented array */
void printindexarray(float *a, int *ind, int n, const char *name){
  std::cout << name << "\t = [";
  for(int i = 0; i < n; ++i){
    std::cout << a[ind[i]] << ", ";
  }
  std::cout << "]\n";
}

/* print indexed fragmented array */
void printindexarrayfrag(float *a, findex* ind, int m, const char *name){
  std::cout << name << "\t = [";
  for(int j = 0; j < m; ++j){
    std::cout << a[ind[j].i] << ", ";
  }
  std::cout << "    ";
  std::cout << "]\n";
}

void printarrayfrag(float *a, int m, const char *name){
  std::cout << name << "\t = [";
  for(int j = 0; j < m; ++j) {
    std::cout << a[j] << ", ";
  }
  printf("]\n");
}

void reset_array(float *a, int n, float val){
  for(int i=0; i<n; ++i){
    a[i] = val;
  }
}

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

