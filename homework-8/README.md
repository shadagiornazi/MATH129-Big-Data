# Homework 8

K-means clustering, the elbow method, and Non-Negative Matrix Factorization (NMF) for topic modeling.

## Files
- `hw8pr1.py`: Python implementation of K-means clustering
- `5000_points.csv`: Dataset containing 5,000 two-dimensional observations
- `kmeans_1.png`: Clustering cost versus number of clusters
- `kmeans_2.png`: Visualization of five clusters and their centroids
- `nmf_als.py`: NMF implementation using Lee and Seung's multiplicative-update algorithm (extra credit)
- `nmf_cvg.png`: NMF reconstruction error versus iteration
- `nmf_topics.txt`: Top 20 words for each of the 20 discovered topics

## Results

### Problem 1: K-means Clustering

K-means was evaluated for k = 1 through 20 using five random initializations per cluster count.

- **Suggested number of clusters:** k = 5 (elbow heuristic)
- **Cost at k = 5:** 6.8042 × 10⁸
- **Minimum observed cost:** 1.6139 × 10⁸ at k = 20

The elbow heuristic suggests five clusters as a reasonable compromise between clustering cost and model complexity.

**Implementation note:** Standard K-means minimizes squared Euclidean distances using arithmetic mean updates. This implementation uses the standard mean updates but reports the sum of unsquared Euclidean distances as its clustering cost. Therefore, the reported cost is not the exact objective minimized by the standard K-means updates.

### Extra Credit 1: Non-Negative Matrix Factorization

NMF was applied to 10,788 Reuters news documents using TF-IDF representations with 10,000 features and 20 latent topics.

- **Initial reconstruction error:** 10,757.7992
- **Final reconstruction error:** 8,790.2183
- **Error reduction:** 18.29%
- **Iterations:** 100
- **Objective non-increasing:** True
- **Factor matrices nonnegative:** True

The reconstruction error decreased throughout optimization, with the largest improvements occurring during the first 20 iterations. The model identified recognizable topics related to corporate finance, international trade, inflation, banking, crude oil, and agricultural commodities.

## Methods

- **K-means:** Iterative nearest-centroid assignment and arithmetic mean updates
- **Cluster selection:** Elbow method
- **NMF:** Lee and Seung's multiplicative-update algorithm
- **Text representation:** TF-IDF

## Libraries

NumPy, pandas, Matplotlib, NLTK, SciPy, and scikit-learn.
