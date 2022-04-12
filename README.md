# Expectation-Maximization-algorithm
The expectation-maximization algorithm is a widely applicable method for iterative computation of maximum likelihood estimates. The K-means algorithm is the most famous variant of this algorithm.

At every iteration of this algorithm, two steps are followed, known as Expectation step (E-Step) and the Maximization step (M-step), and hence the algorithm named as EM algorithm:

1. Start with an arbitrary initial choice of parameters (mean, variance) for each cluster.
2. (Expectation) Compute the probability of each possible sample, given the parameters.
3. (Maximization) Use the just-computed probabilities of each sample to compute a better estimate for the parameters.
4. Repeat steps 2 and 3 to convergence.

With the program, it is possible to explain and visualize the steps followed by the algorithm using synthetic 2D data. The program allows the adjustment of the data in its distribution and number of clusters.
