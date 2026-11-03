# Homework 8

K-means clustering and cluster selection using the elbow method.

## Files
- `hw8pr1.py`: Python implementation of K-means clustering
- `5000_points.csv`: Dataset containing 5,000 two-dimensional observations
- `kmeans_1.png`: Clustering cost versus number of clusters
- `kmeans_2.png`: Visualization of the selected clusters and centroids

## Results

K-means was evaluated for cluster counts from 1 to 20 using five random initializations per cluster count.

- **Suggested number of clusters:** k = 5 (elbow heuristic)
- **Cost at k = 5:** 6.8042 × 10⁸
- **Minimum observed cost:** 1.6139 × 10⁸ at k = 20

The elbow heuristic suggests five clusters as a reasonable compromise between model complexity and clustering cost.

## Implementation Note

Standard K-means minimizes the sum of squared Euclidean distances:

\[
J_{\text{K-means}}=\sum_{i=1}^{m}\|x_i-\mu_{c_i}\|_2^2
\]

The implementation uses standard K-means assignment and mean-update steps but evaluates clustering quality using the sum of unsquared Euclidean distances:

\[
J_{\text{reported}}=\sum_{i=1}^{m}\|x_i-\mu_{c_i}\|_2
\]

Thus, the reported cost differs from the objective directly minimized by standard K-means. The unsquared-distance cost is used to compare cluster counts, while the centroids are updated using arithmetic means.
