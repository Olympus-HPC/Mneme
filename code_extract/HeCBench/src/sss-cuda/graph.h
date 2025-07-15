#define GRAPH_H

typedef class Graph *LPGraph;

class Graph {
public:
  // data members
  myInt nVertices;
  myInt *d_nVertices; // number of vertices in the graph
  myInt **Edge;
  int *d_EdgeField; // matrix containing the edges of the graph

  myInt *Labels;
  myInt *d_Labels; // identifies the connected components of the graph
  myInt nLabels;
  myInt *d_nLabels;     // number of labels or connected components
  myInt **Cliques;      // storage for cliques
  myInt *CliquesDimens; // number of vertices in each clique
  myInt nCliques;
  myInt *d_nCliques; // number of cliques

public:
  myInt *TreeEdgeA; // edges of the clique tree
  myInt *TreeEdgeB;
  myInt nTreeEdges; // number of edges in the generated clique tree
public:
  myInt **Separators; // storage for separators
  myInt *SeparatorsDimens;
  myInt nSeparators;
  // private:
  myInt *localord;

  // methods
public:
  Graph();                     // constructor
  Graph(LPGraph InitialGraph); // constructor
  ~Graph();                    // destructor
public:
  myInt SearchVertex();       // identifies the next vertex to be eliminated
  void FlipEdge(myInt which); // flips an edge on a graph which is a myInteger
                              // between 0 and the total number of edges

public:
  // the MSS (Minimal Sufficient Statistics) are the maximal cliques for our
  // graph
  void InitGraph(myInt n);
  void CopyGraph(LPGraph G);
  void GenerateCliques(myInt label);
  myInt CheckCliques(myInt start,
                     myInt end); // checks whether each generated component is
                                 // complete in the given graph
  myInt IsClique(
      myInt *vect,
      myInt nvect); // checks if the vertices in vect form a clique in our graph

  void GenerateSeparators();
  void AttachLabel(myInt v, myInt label);
  void GenerateLabels();
  myInt GenerateAllCliques();
  myInt IsDecomposable();
  myInt IfDecomposable();

  myInt CanDeleteEdge(myInt a, myInt b);
  bool CanAddEdge(myInt a, myInt b);

  Real ScoreDeleteEdge(myInt a, myInt b, myInt which_ab, Real *D_prior,
                       Real *D_post, myInt delta, myInt n_sub, Real score,
                       int nEdges);
  Real ScoreAddEdge(myInt a, myInt b, Real *D_prior, Real *D_post, myInt delta,
                    myInt n_sub, Real score, int nEdges);
};

//////////////////////////////////////////////////////////////////////

typedef class SectionGraph *LPSectionGraph;

class SectionGraph : public Graph {
public:
  myInt *
      Eliminated; // shows which vertices were eliminated from the initial graph
  myInt nEliminated; // number of vertices we eliminated

  // methods
public:
  SectionGraph(LPGraph InitialGraph, myInt *velim); // constructor
  ~SectionGraph();                                  // destructor

public:
  myInt IsChain(myInt u, myInt v); // see if there is a chain between u and v
                                   // or, equivalently, checks if u and v are in
                                   // the same connected component
};

////////////////////////////////////////////////////////////////////////

typedef class EliminationGraph *LPEliminationGraph;

class EliminationGraph : public Graph {
public:
  myInt *
      Eliminated; // shows which vertices were eliminated from the initial graph
  myInt nEliminated; // number of vertices we eliminated

  // methods
public:
  EliminationGraph(LPGraph InitialGraph, myInt vertex); // constructor
  ~EliminationGraph();                                  // destructor
public:
  myInt SearchVertex(); // identify a vertex to be eliminated
public:
  void EliminateVertex(myInt x); // eliminates an extra vertex
};

//////////////////////////////////////////////////////////////////////////

// constructs the minimum fill-in graph for a nondecomposable graph
void TurnFillInGraph(LPGraph graph);