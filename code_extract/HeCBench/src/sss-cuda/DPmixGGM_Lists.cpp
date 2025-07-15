#define LISTS_CPP
#ifndef GRAPH_CPP
#include "graph.cpp"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif
#ifndef DPMIXGGM_CPP
#include "DPmixGGM.cpp"
#endif

typedef class DPmixGGMlist *List;

class DPmixGGMlist {
  // variables
public:
  int n;
  int p;
  myInt M;

  myInt *L_list;
  myInt *xi_list;
  myInt *edge_list;
  Real *score_list;

  myInt ind_minOfListScores;
  Real minOfListScores;

  // functions
public:
  DPmixGGMlist(myInt size, int n, int p);
  // DPmixGGMlist (myInt size, State a);
  ~DPmixGGMlist();

  void FlushList(State a);
  void WriteList(ofstream &out);
  void ReadFromList(State a, myInt m);

  void UpdateList(myInt L, myInt *xi, LPGraph *graphlist, Real score);
  void UpdateList(State a);

  LPGraph ProposeGraph(myInt NN_xi, myInt *which_xi, Real sFactor);
};

DPmixGGMlist::DPmixGGMlist(myInt size, int in_n, int in_p) {
  int i;
  n = in_n;
  p = in_p;
  M = size;
  int ee = p * (p - 1) / 2;

  L_list = new myInt[M];
  for (i = 0; i < M; i++) {
    L_list[i] = -1;
  }
  xi_list = new myInt[M * n];
  for (i = 0; i < M * n; i++) {
    xi_list[i] = -1;
  }
  edge_list = new myInt[M * n * ee];
  for (i = 0; i < (M * n * ee); i++) {
    edge_list[i] = -1;
  }
  score_list = new Real[M];
  for (i = 0; i < M; i++) {
    score_list[i] = NEG_INF;
  }

  ind_minOfListScores = 0;
  minOfListScores = NEG_INF;
}

DPmixGGMlist::~DPmixGGMlist() {
  delete[] L_list;
  delete[] xi_list;
  delete[] edge_list;
  delete[] score_list;
}

void DPmixGGMlist::FlushList(State a) {
  int i;
  int ee = p * (p - 1) / 2;

  for (i = 0; i < M; i++) {
    L_list[i] = -1;
  }
  for (i = 0; i < M * n; i++) {
    xi_list[i] = -1;
  }
  for (i = 0; i < (M * n * ee); i++) {
    edge_list[i] = -1;
  }
  for (i = 0; i < M; i++) {
    score_list[i] = NEG_INF;
  }

  int q, r, l;
  L_list[0] = a->L;
  for (i = 0; i < n; i++) {
    xi_list[i] = a->xi[i];
  }
  for (i = 0; i < n; i++) {
    l = 0;
    for (q = 0; q < p - 1; q++) {
      for (r = q + 1; r < p; r++) {
        edge_list[i * ee + l] = a->graphlist[a->xi[i]]->Edge[q][r];
        l++;
      }
    }
  }
  score_list[0] = a->plp;
  for (i = 0; i < (a->L); i++) {
    score_list[0] += a->pll[i];
  }

  ind_minOfListScores = 1;
  minOfListScores = NEG_INF;
}

void DPmixGGMlist::WriteList(ofstream &out) {
  int i, j, l, L, q, r, t;
  int ee = p * (p - 1) / 2;

  for (t = 0; t < M; t++) {
    if (L_list[t] == -1) {
      continue;
    }

    L = L_list[t];
    out << L << " " << score_list[t] << " ";
    for (i = 0; i < n; i++) {
      out << xi_list[t * n + i] << " ";
    }

    LPGraph *graphlist = new LPGraph[L];
    for (l = 0; l < L; l++) {
      graphlist[l] = new Graph();
      graphlist[l]->InitGraph(p);
    }
    for (l = 0; l < L; l++) {
      for (i = 0; i < n; i++) {
        if (xi_list[t * n + i] == l)
          break;
      }
      j = 0;
      for (q = 0; q < p - 1; q++) {
        graphlist[l]->Edge[q][q] = 0;
        for (r = q + 1; r < p; r++) {
          graphlist[l]->Edge[q][r] = edge_list[t * n * ee + i * ee + j];
          graphlist[l]->Edge[r][q] = graphlist[l]->Edge[q][r];
          out << graphlist[l]->Edge[q][r] << " ";
          j++;
        }
      }
      graphlist[l]->Edge[p - 1][p - 1] = 0;
      graphlist[l]->GenerateAllCliques();
    }

    for (l = 0; l < L; l++) {
      out << graphlist[l]->nCliques << " ";
      for (i = 0; i < (graphlist[l]->nCliques); i++) {
        out << graphlist[l]->CliquesDimens[i] << " ";
        for (j = 0; j < (graphlist[l]->CliquesDimens[i]); j++) {
          out << graphlist[l]->Cliques[i][j] << " ";
        }
      }

      out << graphlist[l]->nTreeEdges << " ";
      for (i = 0; i < (graphlist[l]->nTreeEdges); i++) {
        out << graphlist[l]->TreeEdgeA[i] << " " << graphlist[l]->TreeEdgeB[i]
            << " ";
      }

      out << graphlist[l]->nSeparators << " ";
      for (i = 0; i < (graphlist[l]->nSeparators); i++) {
        out << graphlist[l]->SeparatorsDimens[i] << " ";
        for (j = 0; j < (graphlist[l]->SeparatorsDimens[i]); j++) {
          out << graphlist[l]->Separators[i][j] << " ";
        }
      }
    }

    out << endl;
    for (l = 0; l < L; l++) {
      delete graphlist[l];
    };
    delete[] graphlist;
  }
}

void DPmixGGMlist::ReadFromList(State a, myInt m) {
  int i, j, k, l, L, t;
  int ee = p * (p - 1) / 2;

  for (l = 0; l < (a->L); l++) {
    delete a->graphlist[l];
  };
  delete[] a->graphlist;
  delete[] a->pll; // delete previous memory spaces
  L = L_list[m];
  a->L = L;
  a->pll = new Real[L];
  a->graphlist = new LPGraph[L];
  for (l = 0; l < L; l++) {
    a->graphlist[l] = new Graph();
    a->graphlist[l]->InitGraph(p);
  }
  for (i = 0; i < n; i++) {
    a->xi[i] = xi_list[m * n + i];
  }
  for (l = 0; l < L; l++) {
    for (k = 0; k < n; k++) {
      if (a->xi[k] == l) {
        break;
      }
    };
    t = 0;
    for (i = 0; i < (p - 1); i++) {
      for (j = (i + 1); j < p; j++) {
        a->graphlist[l]->Edge[i][j] = edge_list[m * n * ee + k * ee + t];
        a->graphlist[l]->Edge[j][i] = a->graphlist[l]->Edge[i][j];
        t++;
      }
    }
    a->graphlist[l]->GenerateAllCliques();
  }
}

// OVERWRITES A GIVEN LIST OF MODELS WITH A GIVEN MODEL IF IT HAS HIGHER SCORE
void DPmixGGMlist::UpdateList(myInt L, myInt *xi, LPGraph *graphlist,
                              Real score) {
  int i, j, k, l, q, r, t;
  int ee = p * (p - 1) / 2;

  // check if the candidate model deserve to be in the list of models -- if so,
  // do the necessary
  bool flag;
  int count;
  myInt *xi_new = new myInt[n];
  myInt *index_set = new myInt[L];
  for (i = 0; i < n; i++) {
    xi_new[i] = -1;
  }

  if (score > minOfListScores) {
    // reindexing (lexicographically) the incoming model -- if there is a
    // redundant index, getting rid of it
    myInt xi_temp = xi[0];
    index_set[0] = xi_temp;
    for (i = 0; i < n; i++) {
      if (xi[i] == xi_temp) {
        xi_new[i] = 0;
      }
    }

    j = 1;
    l = 0;
    while (j < n) {
      k = j;
      l++;
      while (k < n) // finding a new cluster index
      {
        flag = 1;
        for (t = 0; t < l; t++) {
          if (xi[k] == index_set[t]) {
            flag = 0;
            break;
          }
        }
        if (flag) {
          xi_temp = xi[k];
          index_set[l] = xi_temp;
          for (i = 0; i < n; i++) {
            if (xi[i] == xi_temp) {
              xi_new[i] = l;
            }
          }
          break;
        } else {
          k++;
        }
      }
      j = k + 1;
    }
    k = l;

    // check if this model is already in the list of models
    count = 0;
    for (t = 0; t < M; t++) {
      flag = 0;
      if (L_list[t] != k) {
        flag = 1;
      }

      if (flag == 0) {
        for (i = 0; i < n; i++) {
          if (xi_list[t * n + i] != xi_new[i]) {
            flag = 1;
            break;
          }
        }
      };

      if (flag == 0) {
        for (l = 0; l < k; l++) {
          for (i = 0; i < n; i++) {
            if (xi_list[t * n + i] == l)
              break;
          } // finiding an observation with cluster index l

          j = 0;
          for (q = 0; q < p - 1; q++) {
            for (r = q + 1; r < p; r++) {
              if (edge_list[t * n * ee + i * ee + j] !=
                  graphlist[xi[i]]->Edge[q][r]) {
                flag = 1;
                break;
              };
              j++;
            }
            if (flag)
              break;
          }
          if (flag)
            break;
        }
      }

      if (flag)
        count++;
    }

    // if the model is not in list, substitute the lowest scoring model with the
    // candidate
    if (count == M) {
      L_list[ind_minOfListScores] = k; // cout << "effective L = " << k << endl;
      for (i = 0; i < n; i++) {
        xi_list[ind_minOfListScores * n + i] = xi_new[i];
        t = 0;
        for (q = 0; q < p - 1; q++) {
          for (r = q + 1; r < p; r++) {
            edge_list[ind_minOfListScores * n * ee + i * ee + t] =
                graphlist[xi[i]]->Edge[q][r];
            t++;
          }
        }
      }
      score_list[ind_minOfListScores] = score;

      // searching for minimum scoring model in the list
      ind_minOfListScores = 0;
      minOfListScores = score_list[0];
      for (t = 1; t < M; t++) {
        if (score_list[t] < minOfListScores) {
          minOfListScores = score_list[t];
          ind_minOfListScores = t;
        }
      }
    }
  }

  delete[] xi_new;
  delete[] index_set;
}

void DPmixGGMlist::UpdateList(State a) {
  Real score = a->plp;
  for (myInt i = 0; i < a->L; i++) {
    score += a->pll[i];
  }
  UpdateList(a->L, a->xi, a->graphlist, score);
}

// PROPOSES A DECOMPOSABLE GRAPH
LPGraph DPmixGGMlist::ProposeGraph(myInt NN_xi, myInt *which_xi, Real sFactor) {
  int i, j, k, l, m, t;
  Real a, s;
  bool temp;
  int ee = p * (p - 1) / 2;

  LPGraph newgraph = new Graph();
  newgraph->InitGraph(p);

  // feature extraction
  l = 0;
  for (i = 0; i < p - 1; i++) {
    newgraph->Edge[i][i] = 0;
    for (j = i + 1; j < p; j++) {
      s = NN_xi * M * sFactor;
      a = s;

      for (m = 0; m < M; m++) {
        for (k = 0; k < NN_xi; k++) {
          t = edge_list[m * n * ee + which_xi[k] * ee + l];
          if (t != -1) {
            a += Real((bool)t);
            s += 1.0;
          }
        }
      }
      l++;

      temp = (gsl_ran_flat(rnd, 0.0, 1.0) < a / s);
      newgraph->Edge[i][j] = temp;
      newgraph->Edge[j][i] = temp;
    }
  }
  newgraph->Edge[p - 1][p - 1] = 0;
  TurnFillInGraph(newgraph);
  newgraph->GenerateAllCliques();

  return newgraph;
}