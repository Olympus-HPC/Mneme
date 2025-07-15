#!/usr/bin/env python

import networkx as nx

#nx.write_edgelist(nx.DiGraph(nx.complete_graph(100)), 'complete-small.txt', data=False)
#nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(100, 3)), 'social-small.txt', data=False)
#nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(200000, 3)), 'social-large-200k.txt', data=False)
#nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(400000, 3)), 'social-large-400k.txt', data=False)
nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(800000, 3)), 'social-large-800k.txt', data=False)
#nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(1600000, 3)), 'social-large-1600k.txt', data=False)
#nx.write_edgelist(nx.DiGraph(nx.barabasi_albert_graph(3200000, 3)), 'social-large-3200k.txt', data=False)
