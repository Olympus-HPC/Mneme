#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRAPH_CPP
#ifndef GRAPH_H
#include "graph.h"
#endif
#ifndef GWISH_CPP
#include "gwish.cpp"
#endif

// class Graph::Begins
Graph::Graph() {
  nVertices = 0;
  Edge = NULL;
  Labels = NULL;
  nLabels = 0;
  Cliques = NULL;
  CliquesDimens = NULL;
  nCliques = 0;
  TreeEdgeA = NULL;
  TreeEdgeB = NULL;
  nTreeEdges = 0;
  Separators = NULL;
  SeparatorsDimens = NULL;
  nSeparators = 0;
  localord = NULL;
  return;
}

Graph::Graph(LPGraph InitialGraph) {
  nVertices = 0;
  Edge = NULL;
  Labels = NULL;
  nLabels = 0;
  Cliques = NULL;
  CliquesDimens = NULL;
  nCliques = 0;
  TreeEdgeA = NULL;
  TreeEdgeB = NULL;
  nTreeEdges = 0;
  Separators = NULL;
  SeparatorsDimens = NULL;
  nSeparators = 0;
  localord = NULL;

  ///////////////////////////////////////
  myInt i, j;
  InitGraph(InitialGraph->nVertices);
  for (i = 0; i < nVertices; i++) {
    for (j = 0; j < nVertices; j++) {
      Edge[i][j] = InitialGraph->Edge[i][j];
    }
  }

  return;
}

Graph::~Graph() {
  myInt i; // cout << "-> "; fflush(stdout);

  for (i = 0; i < nVertices; i++) {
    delete[] Edge[i];
    Edge[i] = NULL;
  }
  delete[] Edge;
  Edge = NULL;

  delete[] Labels;
  Labels = NULL;

  delete[] Cliques[0]; // for(i=0; i<nVertices; i++) { delete[] Cliques[i];
                       // Cliques[i] = NULL; }
  delete[] Cliques;
  Cliques = NULL;
  delete[] CliquesDimens;
  CliquesDimens = NULL;

  delete[] TreeEdgeA;
  TreeEdgeA = NULL;
  delete[] TreeEdgeB;
  TreeEdgeB = NULL;

  delete[] Separators[0]; // for(i=0; i<nVertices; i++) { delete[]
                          // Separators[i]; Separators[i] = NULL; }
  delete[] Separators;
  Separators = NULL;
  delete[] SeparatorsDimens;
  SeparatorsDimens = NULL;

  delete[] localord;
  localord = NULL;

  return;
}

void Graph::InitGraph(myInt n) {
  myInt i;

  // memory initialised to zeros when necessary
  nVertices = n;

  Edge = new myInt *[nVertices];
  for (i = 0; i < n; i++) {
    Edge[i] = new myInt[nVertices];
  }

  nLabels = 0;
  Labels = new myInt[nVertices];

  nCliques = 0;

  Cliques = new myInt *[nVertices];
  Cliques[0] = new myInt[nVertices * nVertices];
  for (i = 1; i < n; i++) {
    Cliques[i] = Cliques[i - 1] + nVertices;
  }
  CliquesDimens = new myInt[nVertices];

  nTreeEdges = 0;
  TreeEdgeA = new myInt[nVertices];
  TreeEdgeB = new myInt[nVertices];

  Separators = new myInt *[nVertices];
  Separators[0] = new myInt[nVertices * nVertices];
  for (i = 1; i < n; i++) {
    Separators[i] = Separators[i - 1] + nVertices;
  }
  SeparatorsDimens = new myInt[nVertices];

  localord = new myInt[nVertices];

  return;
}

void Graph::CopyGraph(LPGraph G) {
  int i, j;
  myInt n = G->nVertices;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      Edge[i][j] = G->Edge[i][j];
    }
  }
  nLabels = G->nLabels;
  for (i = 0; i < n; i++) {
    Labels[i] = G->Labels[i];
  }
  nCliques = G->nCliques;
  for (i = 0; i < n; i++) {
    CliquesDimens[i] = G->CliquesDimens[i];
    for (j = 0; j < n; j++) {
      Cliques[i][j] = G->Cliques[i][j];
    }
  }
  nTreeEdges = G->nTreeEdges;
  for (i = 0; i < n; i++) {
    TreeEdgeA[i] = G->TreeEdgeA[i];
    TreeEdgeB[i] = G->TreeEdgeB[i];
  }
  nSeparators = G->nSeparators;
  for (i = 0; i < n; i++) {
    SeparatorsDimens[i] = G->SeparatorsDimens[i];
    for (j = 0; j < n; j++) {
      Separators[i][j] = G->Separators[i][j];
    }
  }
  for (i = 0; i < n; i++) {
    localord[i] = G->localord[i];
  }
}

void Graph::GenerateCliques(myInt label) {
  myInt i, j, k, p, r;
  myInt n = nVertices;
  myInt *clique = new myInt[nVertices];
  memset(clique, 0, nVertices * sizeof(myInt));

  myInt countA, countB;
  myInt *listA = new myInt[n];
  myInt *listB2 = new myInt[n];

  // clean memory
  memset(localord, 0, nVertices * sizeof(myInt));

  myInt v, vk;
  myInt PrevCard = 0;
  myInt NewCard;
  myInt s = nCliques - 1; // cout << "s = " << s << endl;

  countA = 0;
  for (i = 0; i < n; i++) {
    if (Labels[i] == label) {
      listA[countA] = i;
      countA++;
    }
  };
  countB = 0;

  for (i = n; i >= 0; i--) {
    NewCard = -1;

    // choose a vertex v...
    for (j = 0; j < countA; j++) {
      myInt maxj = 0;
      for (r = 0; r < countB; r++) {
        if (Edge[listA[j]][listB2[r]]) {
          maxj++;
        }
      }
      if (maxj > NewCard) {
        v = listA[j];
        NewCard = maxj;
      }
    }

    // printf("i=%d, NewCard=%d, PrevCard=%d countA=%d,
    // countB=%d\n",i,NewCard,PrevCard,countA,countB);

    if (NewCard == -1) {
      break;
    }

    localord[v] = i;
    if (NewCard <= PrevCard) { // begin new clique
      s++;

      for (r = 0; r < countB; r++) {
        if (Edge[v][listB2[r]]) {
          Cliques[s][CliquesDimens[s]] = listB2[r];
          CliquesDimens[s]++;
        }
      }

      if (NewCard != 0) { // get edge to parent
        vk = Cliques[s][0];
        k = localord[vk]; // cout << "(" << clique[Cliques[s][0]] << " ";
        for (r = 1; r < CliquesDimens[s]; r++) {
          if (localord[Cliques[s][r]] < k) {
            vk = Cliques[s][r];
            k = localord[vk];
          }; // cout << clique[Cliques[s][r]] << " ";
        }
        // cout << "| ";
        // for(r=0; r<CliquesDimens[s]; r++) { cout << Cliques[s][r] << " " <<
        // localord[Cliques[s][r]] << " : "; }; cout << endl;

        p = clique[vk];
        TreeEdgeA[nTreeEdges] = s;
        TreeEdgeB[nTreeEdges] = p;
        nTreeEdges++; // cout << s << "-" << p << ") ";
      }
    }
    clique[v] = s;
    Cliques[s][CliquesDimens[s]] = v;
    CliquesDimens[s]++;

    for (j = 0; j < countA; j++) {
      if (listA[j] == v) {
        break;
      }
    };
    listA[j] = listA[countA - 1];
    countA--;
    listB2[countB] = v;
    countB++;
    PrevCard = NewCard;
  }

  nCliques = s + 1;

  delete[] clique;
  delete[] listA;
  delete[] listB2;
  return;
}

myInt Graph::CheckCliques(myInt start, myInt end) {
  myInt i, j, k;

  for (i = start; i < end; i++) {
    for (j = 0; j < CliquesDimens[i] - 1; j++) {
      for (k = j + 1; k < CliquesDimens[i]; k++) {
        if (Edge[Cliques[i][j]][Cliques[i][k]] == 0) {
          return (-i - 1);
        }
      }
    }
  }

  return 1;
}

void Graph::GenerateSeparators() {
  myInt i, j, k;
  myInt FirstClique, SecondClique;
  myInt v;

  for (i = 0; i < nTreeEdges; i++) {
    FirstClique = TreeEdgeA[i];
    SecondClique = TreeEdgeB[i];

    for (j = 0; j < CliquesDimens[FirstClique]; j++) {
      v = Cliques[FirstClique][j];
      for (k = 0; k < CliquesDimens[SecondClique]; k++) {
        if (v == Cliques[SecondClique][k]) {
          Separators[i][SeparatorsDimens[i]] = v;
          SeparatorsDimens[i]++;
          break;
        }
      }
    }
  }

  nSeparators = nTreeEdges;
  return;
}

void Graph::AttachLabel(myInt v, myInt label) {
  myInt i;

  // only if v has not been labeled yet
  if (Labels[v] == 0) {
    Labels[v] = label;
    for (i = 0; i < nVertices; i++) {
      if (Edge[v][i] == 1) {
        AttachLabel(i, label);
      }
    }
  }
  return;
}

void Graph::GenerateLabels() {
  myInt i;
  myInt NotFinished = 1;
  myInt label = 0;
  myInt v;

  nLabels = 0;
  memset(Labels, 0, nVertices * sizeof(myInt));

  while (NotFinished) {
    v = -1;
    for (i = 0; i < nVertices; i++) {
      if (Labels[i] == 0) {
        v = i;
        break;
      }
    }

    if (v == -1) {
      NotFinished = 0;
    } else {
      label++;
      AttachLabel(v, label);
    }
  }
  nLabels = label;
  return;
}

myInt Graph::GenerateAllCliques() {
  myInt i;
  myInt n = nVertices;
  myInt label;
  myInt start;
  GenerateLabels();

  // clean memory for cliques
  nCliques = 0;
  for (i = 0; i < n; i++) {
    memset(Cliques[i], 0, n * sizeof(myInt));
  };
  memset(CliquesDimens, 0, n * sizeof(myInt));
  nTreeEdges = 0;
  memset(TreeEdgeA, 0, n * sizeof(myInt));
  memset(TreeEdgeB, 0, n * sizeof(myInt));

  for (label = 1; label <= nLabels; label++) {
    start = nCliques;
    GenerateCliques(label);
    if (CheckCliques(start, nCliques) < 0) {
      return 0;
    } // this is not a decomposable model
  }

  nSeparators = 0;
  for (i = 0; i < n; i++) {
    memset(Separators[i], 0, n * sizeof(myInt));
  };
  memset(SeparatorsDimens, 0, n * sizeof(myInt));
  GenerateSeparators();
  return 1;
}

// Just does not generate separators
myInt Graph::IfDecomposable() {
  myInt i;
  myInt n = nVertices;
  myInt label;
  myInt start;

  GenerateLabels(); // cout << "nLabels = " << nLabels << endl;

  // clean memory for cliques
  nCliques = 0;
  for (i = 0; i < n; i++) {
    memset(Cliques[i], 0, n * sizeof(myInt));
  };
  memset(CliquesDimens, 0, n * sizeof(myInt));
  nTreeEdges = 0;
  memset(TreeEdgeA, 0, n * sizeof(myInt));
  memset(TreeEdgeB, 0, n * sizeof(myInt));

  for (label = 1; label <= nLabels; label++) {
    start = nCliques; // cout << "label = " << label << ", start = " << start <<
                      // " " << endl;
    GenerateCliques(label); // cout << "nCliques = " << nCliques << endl;
    if (CheckCliques(start, nCliques) < 0) {
      return 0;
    } // this is not a decomposable model
  }

  nSeparators = 0;
  for (i = 0; i < n; i++) {
    memset(Separators[i], 0, n * sizeof(myInt));
  };
  memset(SeparatorsDimens, 0, n * sizeof(myInt));
  // GenerateSeparators();
  return 1;
}

myInt Graph::SearchVertex() {
  myInt x, u, v;
  myInt okay;
  myInt *sxAdj = new myInt[nVertices];
  memset(sxAdj, 0, nVertices * sizeof(myInt));

  for (x = 0; x < nVertices; x++) {
    memmove(sxAdj, Edge[x], nVertices * sizeof(myInt));
    sxAdj[x] = 1;
    okay = 1;
    for (u = 0; u < nVertices; u++) {
      if ((u != x) && (Edge[x][u] == 1)) {
        sxAdj[u] = 0; // we take u out
        for (v = u + 1; v < nVertices; v++) {
          if ((v != x) && (Edge[x][v] == 1) && (Edge[u][v] == 0)) {
            sxAdj[v] = 0; // we take v out
            SectionGraph sgraph(this, sxAdj);
            okay = sgraph.IsChain(u, v);
            sxAdj[v] = 1; // now put v back in the adjacency list of x
          }
          if (!okay)
            break;
        }
        sxAdj[u] = 1; // we put u back
      }
      if (!okay)
        break;
    }
    if (okay)
      break;
  }
  delete[] sxAdj;
  if (x == nVertices)
    x = -1;
  return x;
}

void Graph::FlipEdge(myInt which) {
  myInt i, j, k = 0;
  myInt p = nVertices;

  for (i = 0; i < p - 1; i++) {
    for (j = i + 1; j < p; j++) {
      if (k == which) {
        Edge[i][j] = 1 - Edge[i][j];
        Edge[j][i] = 1 - Edge[j][i];
        return;
      }
      k++;
    }
  }
}

myInt Graph::IsDecomposable() { return GenerateAllCliques(); }
// class Graph::Ends

// class SectionGraph::Begins
SectionGraph::SectionGraph(LPGraph InitialGraph, myInt *velim)
    : Graph(InitialGraph) {
  myInt i, j;

  Eliminated = new myInt[nVertices];
  memset(Eliminated, 0, nVertices * sizeof(myInt));
  nEliminated = 0;
  for (i = 0; i < nVertices; i++) {
    if (velim[i]) {
      Eliminated[i] = 1;
      nEliminated++;
    }
  }
  // delete all the edges corresponding to the vertices we eliminated
  for (i = 0; i < nVertices; i++) {
    if (Eliminated[i]) {
      for (j = 0; j < nVertices; j++) {
        if (1 == Edge[i][j]) {
          Edge[i][j] = Edge[j][i] = 0;
        }
      }
    }
  }
  return;
}

SectionGraph::~SectionGraph() {
  delete[] Eliminated;
  nEliminated = 0;
  return;
}

myInt SectionGraph::IsChain(myInt u, myInt v) {
  if (nLabels == 0) {
    GenerateLabels();
  }
  if (Eliminated[u] || Eliminated[v]) {
    printf("One of the vertices %d,%d has been eliminated...\n", u, v);
    exit(1);
  }
  if (Labels[u] == Labels[v])
    return 1;
  return 0;
}
// class SectionGraph::Ends

// class EliminationGraph::Begins
EliminationGraph::EliminationGraph(LPGraph InitialGraph, myInt vertex)
    : Graph(InitialGraph) {
  Eliminated = new myInt[nVertices]; // CheckPomyInter(Eliminated);
  memset(Eliminated, 0, nVertices * sizeof(myInt));
  nEliminated = 0;
  EliminateVertex(vertex);
  return;
}

EliminationGraph::~EliminationGraph() {
  delete[] Eliminated;
  nEliminated = 0;
  return;
}

void EliminationGraph::EliminateVertex(myInt x) {
  myInt i, j;

  // adding edges in Def(Adj(x)) so that Adj(x) becomes a clique
  for (i = 0; i < nVertices; i++) {
    if ((i != x) && (!Eliminated[i]) && (Edge[x][i] == 1)) {
      for (j = i + 1; j < nVertices; j++) {
        if ((j != x) && (!Eliminated[j]) && (Edge[x][j] == 1) &&
            (Edge[i][j] == 0)) {
          Edge[i][j] = Edge[j][i] = 1;
        }
      }
    }
  }

  // eliminate all edges incident to x
  for (i = 0; i < nVertices; i++) {
    if ((i != x) && (!Eliminated[i]) && (Edge[x][i] == 1)) {
      Edge[x][i] = Edge[i][x] = 0;
    }
  }

  // eliminate vertex x
  Eliminated[x] = 1;
  nEliminated++;
  return;
}

myInt EliminationGraph::SearchVertex() {
  myInt x, u, v;
  myInt okay;
  myInt *sxAdj = new myInt[nVertices];
  memset(sxAdj, 0, nVertices * sizeof(myInt));

  for (x = 0; x < nVertices; x++) {
    if (Eliminated[x])
      continue;
    memmove(sxAdj, Edge[x], nVertices * sizeof(myInt));
    sxAdj[x] = 1;
    okay = 1;
    for (u = 0; u < nVertices; u++) {
      if (Eliminated[u])
        continue;
      if ((u != x) && (Edge[x][u] == 1)) {
        sxAdj[u] = 0; // we take u out
        for (v = u + 1; v < nVertices; v++) {
          if (Eliminated[v])
            continue;
          if ((v != x) && (Edge[x][v] == 1) && (Edge[u][v] == 0)) {
            sxAdj[v] = 0; // we take v out
            SectionGraph sgraph(this, sxAdj);
            okay = sgraph.IsChain(u, v);
            sxAdj[v] = 1; // now put v back in the adjacency list of x
          }
          if (!okay)
            break;
        }
        sxAdj[u] = 1; // we put u back
      }
      if (!okay)
        break;
    }
    if (okay)
      break;
  }
  delete[] sxAdj;
  if (x == nVertices)
    x = -1;
  return x;
}
// class EliminationGraph::Ends

void TurnFillInGraph(LPGraph graph) {
  myInt u, v;
  myInt i;

  LPGraph gfill = graph;
  // if the graph is decomposable, there is no need to do anything
  if (gfill->IsDecomposable())
    return;

  myInt v1 = gfill->SearchVertex();
  // add edges to Def(Adj(x)) so that Adj(x) becomes a clique
  for (u = 0; u < gfill->nVertices; u++) {
    if (gfill->Edge[v1][u] == 1) {
      for (v = u + 1; v < gfill->nVertices; v++) {
        if ((gfill->Edge[v1][v] == 1) && (gfill->Edge[u][v] == 0)) {
          gfill->Edge[v][u] = gfill->Edge[u][v] = 1;
        }
      }
    }
  }
  LPEliminationGraph egraph = new EliminationGraph(graph, v1);
  for (i = 1; i < graph->nVertices - 1; i++) {
    v1 = egraph->SearchVertex();
    for (u = 0; u < egraph->nVertices; u++) {
      if (egraph->Eliminated[u])
        continue;
      if (egraph->Edge[v1][u] == 1) {
        for (v = u + 1; v < egraph->nVertices; v++) {
          if (egraph->Eliminated[v])
            continue;
          if ((egraph->Edge[v1][v] == 1) && (egraph->Edge[u][v] == 0)) {
            gfill->Edge[v][u] = gfill->Edge[u][v] = 1;
            // these are the edges that are added to the initial graph
          }
        }
      }
    }
    egraph->EliminateVertex(v1);
  }
  delete egraph;
  return;
}

// -----------------------------------------------------------------------------------------------
// Based on Scott & Carvalho 2008

bool Graph::CanAddEdge(myInt a, myInt b) {
  myInt i, j, k;
  bool canadd = 0;
  int n = (int)nVertices;
  myInt nS = 0, pR, pT;
  myInt *R = new myInt[nTreeEdges];
  myInt *T = new myInt[nTreeEdges];
  myInt *S = new myInt[n];
  myInt common_parent = 0;
  myInt contain_a, contain_b, contain_S;

  if (Labels[a] != Labels[b]) {
    canadd = 1;
  } // else ..
  else {

    nS = 0;
    for (j = 0; j < n; j++) {
      if (Edge[a][j] && Edge[b][j]) {
        S[nS] = j;
        nS++;
      }
    }
    if (nS == 0) {
      canadd = 0;
    }      // else ....
    else { // HERE;

      // find higest-indexed cliques containing (a & S) and (b & S)
      myInt aSi = -1, bSi = -1;
      for (i = 0; i < nCliques; i++) {
        contain_a = 0;
        contain_b = 0;
        contain_S = 0;
        for (j = 0; j < CliquesDimens[i]; j++) {
          if (Cliques[i][j] == a) {
            contain_a = 1;
          };
          if (Cliques[i][j] == b) {
            contain_b = 1;
          }
          for (k = 0; k < nS; k++) {
            if (Cliques[i][j] == S[k]) {
              contain_S++;
            }
          }
        }
        if (contain_a && (contain_S == nS)) {
          aSi = i;
        }
        if (contain_b && (contain_S == nS)) {
          bSi = i;
        }
      }

      // find the path from aSi to root
      R[0] = -1;
      pR = -1;
      for (i = 0; i < nTreeEdges; i++) {
        if (TreeEdgeA[i] == aSi) {
          R[0] = i;
          pR = 0;
          break;
        }
      }
      for (i = R[0]; i >= 0; i--) {
        if (TreeEdgeA[i] == TreeEdgeB[R[pR]]) {
          pR++;
          R[pR] = i;
        }
      }

      // find the path from bSi to root
      T[0] = -1;
      pT = -1;
      for (i = 0; i < nTreeEdges; i++) {
        if (TreeEdgeA[i] == bSi) {
          T[0] = i;
          pT = 0;
          break;
        }
      }
      for (i = T[0]; i >= 0; i--) {
        if (TreeEdgeA[i] == TreeEdgeB[T[pT]]) {
          pT++;
          T[pT] = i;
        }
      }

      // find the branching point
      k = (pR < pT) ? pR : pT; // min(pR,pT);
      for (i = 0; i <= k; i++) {
        if (TreeEdgeB[R[pR - i]] == TreeEdgeB[T[pT - i]]) {
          common_parent = i;
        } else {
          break;
        }
      }
      if (k != -1) {
        if (TreeEdgeA[R[pR - common_parent]] ==
            TreeEdgeA[T[pT - common_parent]]) {
          common_parent++;
        }
      }

      // check if S is in the path from R[0] to common_parent
      for (i = 0; i <= (pR - common_parent); i++) {
        if (SeparatorsDimens[R[i]] == nS) {
          contain_S = 0;
          for (j = 0; j < SeparatorsDimens[R[i]]; j++) {
            for (k = 0; k < nS; k++) {
              if (Separators[R[i]][j] == S[k]) {
                contain_S++;
              }
            }
          }
          if (contain_S == nS) {
            canadd = 1;
          }
        }
        if (canadd) {
          break;
        }
      }

      // else: check if S is in the path from T[0] to common_parent
      if (!canadd) {
        for (i = 0; i <= (pT - common_parent); i++) {
          if (SeparatorsDimens[T[i]] == nS) {
            contain_S = 0;
            for (j = 0; j < SeparatorsDimens[T[i]]; j++) {
              for (k = 0; k < nS; k++) {
                if (Separators[T[i]][j] == S[k]) {
                  contain_S++;
                }
              }
            }
            if (contain_S == nS) {
              canadd = 1;
            }
          }
          if (canadd) {
            break;
          }
        }
      }
    }
  }

  delete[] R;
  delete[] T;
  delete[] S;

  return (canadd);
}

Real Graph::ScoreAddEdge(myInt a, myInt b, Real *D_prior, Real *D_post,
                         myInt delta, myInt n_sub, Real score, int nEdges) {
  myInt j;
  int n = (int)nVertices;
  myInt nS = 0;
  myInt *S = new myInt[n];

  nS = 0;
  for (j = 0; j < n; j++) {
    if (Edge[a][j] && Edge[b][j]) {
      S[nS] = j;
      nS++;
    }
  }

  {
    myInt *C = new myInt[n];
    myInt nC;
    Real *sub_D = new Real[n * (n + 1) / 2];
    Real cScore;

    nC = nS + 2;
    for (j = 0; j < nS; j++) {
      C[j] = S[j];
    };
    C[nS] = a;
    C[nS + 1] = b;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(n, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score += cScore;

    nC = nS + 1;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(n, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score -= cScore;

    nC = nS + 1;
    C[nS] = b;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(n, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score -= cScore;

    nC = nS;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(n, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score += cScore; // cout << nS << "->";

    delete[] S;
    delete[] C;
    delete[] sub_D;
    return score;
  }
}

myInt Graph::CanDeleteEdge(myInt a, myInt b) {
  myInt i, j;
  myBool contain_a, contain_b;
  myInt count = 0, which_ab;

  for (i = 0; i < nCliques; i++) {
    contain_a = 0;
    contain_b = 0;
    for (j = 0; j < CliquesDimens[i]; j++) {
      if (Cliques[i][j] == a) {
        contain_a = 1;
      };
      if (Cliques[i][j] == b) {
        contain_b = 1;
      }
    }
    if (contain_a && contain_b) {
      which_ab = i;
      count++;
    }
  }

  if (count == 1) {
    return (which_ab);
  } else {
    return (-1);
  }
}

Real Graph::ScoreDeleteEdge(myInt a, myInt b, myInt which_ab, Real *D_prior,
                            Real *D_post, myInt delta, myInt n_sub, Real score,
                            int nEdges) {
  myInt j;

  {
    myInt p = nVertices;
    myInt *C = new myInt[p];
    myInt nC;
    Real *sub_D = new Real[p * (p + 1) / 2];
    Real cScore;

    nC = 0;
    for (j = 0; j < CliquesDimens[which_ab]; j++) {
      if ((Cliques[which_ab][j] != a) && (Cliques[which_ab][j] != b)) {
        C[nC] = Cliques[which_ab][j];
        nC++;
      }
    };
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(p, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score -= cScore;

    nC = CliquesDimens[which_ab] - 1;
    C[nC - 1] = a;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(p, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score += cScore;

    nC = CliquesDimens[which_ab] - 1;
    C[nC - 1] = b;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(p, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score += cScore;

    nC = CliquesDimens[which_ab];
    C[nC - 2] = a;
    C[nC - 1] = b;
    cScore = 0;
    cScore -= gwish_nc_complete(delta, nC, sub_D, 0);
    make_sub_mat_dbl(p, nC, C, D_post, sub_D);
    cScore += gwish_nc_complete(delta + n_sub, nC, sub_D, 1);
    score -= cScore;

    delete[] C;
    delete[] sub_D;
    return score;
  }
}