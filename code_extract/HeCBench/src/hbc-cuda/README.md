hybrid_BC
=========

Hybrid methods for Parallel Betweenness Centrality on the GPU

The only dependency of this code (other than CUDA itself, of course) is Boost. A few small input examples can be found in the graphs directory. Both DIMACS and SNAP graph formats are accepted (assuming unweighted, undirected graphs). DIMACS files are assumed to end in the ".graph" extension whereas SNAP (edgelist) files are assumed to end in either ".txt" or ".edge" extensions. 

Example command line to run and time the algorithm:

$ ./bc -i ./graphs/breast_coexpress.txt

To do the same and compare to sequential execution on the CPU:

$ ./bc -i ./graphs/breast_coexpress.txt -v

To print BC scores to a file:

$ ./bc -i ./graphs/breast_coexpress.txt --printscores=output.txt

If you use the algorithms in this repository for your own research, please cite our SC14 paper: https://dl.acm.org/citation.cfm?id=2683656

More information on the algorithms used in this repository can be found in the paper itself: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.2926&rep=rep1&type=pdf

Additionally, this work was featured as a research highlight in the August issue of Communications of the ACM! https://cacm.acm.org/magazines/2018/8/229768-accelerating-gpu-betweenness-centrality/fulltext

Adam McLaughlin

Adam27X@gatech.edu
