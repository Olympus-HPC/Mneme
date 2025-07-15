# Mean Shift Clustering
## Brief description

Mean shift is a popular non-parametric clustering technique. It is used when the number of cluster centers is unknown apriori.
Based on kernel density estimation, it aims to discover the modes of a density of samples. With the help of a kernel function, mean shift works by updating each sample to be the mean of the points within a given region. More details can be found here [1].

The average complexity is given by *O(N * N * I)*, were *N* is the number of samples and *I* is the number of iteration.


Mean shift works for Euclidean spaces of arbitrary dimensionality (obviously suffering the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)) and in this implementation this number will be not fixed a-priori. 

dataset.tar.gz compresses the following two files
1. `centroids.csv`: The real centroids (that should match the ones computed by mean shift).
2. `data.csv`: The actual data to be clustered.


- `SIGMA` is the standard deviation of the gaussian used to compute the kernel.
- `RADIUS` is used for determining the neighbors of a point.
- `MIN_DISTANCE` is the minimum (L2 squared) distance between two points to be considered belonging to the same clusters.
- `M` is the number of centroids
- `N` is the number of data points 
- `D` is the dimensionality of each data point or centroid
- `NUM_ITER` is the number of iterations that the algorithm will run through.
- `THREADS` is the number of threads in a block.

---

[1]: https://en.wikipedia.org/wiki/Mean_shift
