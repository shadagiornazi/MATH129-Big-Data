"""
MATH 129: Big Data 
Shada Giornazi 
Homework 8, Problem 1
K-Means Clustering
Starter file for k-means(hw8pr1) of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features
	3) Let k be the number of clusters

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= HELPER FUNCTIONS =============

def k_means_cost(X, clusters, label):
    """Calculate the sum of Euclidean distances."""

    assigned = clusters[np.asarray(label).reshape(-1)]

    cost = np.linalg.norm(X - assigned, axis=1).sum()

    return float(cost)


def k_means(X, k, eps=1e-8, max_iter=300,
            print_freq=50, seed=None):
    """Implement K-means clustering."""

    rng = np.random.default_rng(seed)

    m, n = X.shape

    # Randomly initialize cluster centers
    centers = X[
        rng.choice(m, size=k, replace=False)
    ].astype(float).copy()

    cost_list = []
    label = np.zeros((m, 1), dtype=int)

    t_start = time.time()

    for iteration in range(max_iter):

        prev_clusters = centers.copy()

        # Step 1: Assign points to nearest centers
        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :],
            axis=2
        )

        assignments = np.argmin(distances, axis=1)

        # Step 2: Update cluster centers
        for j in range(k):

            members = X[assignments == j]

            if len(members) > 0:
                centers[j] = members.mean(axis=0)

            else:
                centers[j] = X[rng.integers(m)]

        # Reassign points after updating centers
        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :],
            axis=2
        )

        label = np.argmin(
            distances, axis=1
        ).reshape(-1, 1)

        # Calculate cost
        cost = k_means_cost(X, centers, label)
        cost_list.append(cost)

        if print_freq and (iteration + 1) % print_freq == 0:
            print(
                f'-- Iteration {iteration + 1} '
                f'- cost {cost:.4E}'
            )

        # Check convergence
        if np.linalg.norm(prev_clusters - centers) <= eps:
            break

    t_end = time.time()

    print(
        f'-- k={k}: {iteration + 1} iterations, '
        f'cost {cost:.4E}, '
        f'{t_end - t_start:.2f} seconds'
    )

    return centers, label, cost_list


def elbow_choice(costs):
    """Estimate the elbow of the cost curve."""

    x = np.arange(1, len(costs) + 1, dtype=float)
    y = np.asarray(costs, dtype=float)

    x_norm = (x - x[0]) / (x[-1] - x[0])

    y_norm = (
        (y - y[-1]) /
        max(y[0] - y[-1], 1e-12)
    )

    distances = np.abs(
        y_norm + x_norm - 1
    ) / np.sqrt(2)

    return int(np.argmax(distances[1:-1]) + 2)


# ============= MAIN DRIVER =============

if __name__ == '__main__':

    # ============= STEP 0: LOADING DATA =============

    print('==> Step 0: Loading data...')

    path = (
        Path(__file__).resolve().parent /
        '5000_points.csv'
    )

    X = pd.read_csv(
        path,
        sep=r'\s+',
        header=None
    ).to_numpy(dtype=float)

    print(
        f'Loaded {X.shape[0]} points '
        f'with {X.shape[1]} features.'
    )

    # ============= STEP 1: K-MEANS =============

    print(
        '==> Step 1: Finding optimal '
        'number of clusters...'
    )

    cost_k_list = []
    best_models = {}

    # Test k = 1 through 20
    for k in range(1, 21):

        # Run multiple random initializations
        runs = [
            k_means(
                X,
                k,
                seed=2026 + 100 * k + r,
                print_freq=0
            )
            for r in range(5)
        ]

        # Select the run with the lowest cost
        clusters, label, cost_list = min(
            runs,
            key=lambda result: result[2][-1]
        )

        best_models[k] = (clusters, label)

        cost_k_list.append(cost_list[-1])

        print(
            f'-- Number of clusters: {k} '
            f'- best cost: {cost_list[-1]:.4E}'
        )

    # Determine optimal number of clusters
    min_cost_k = int(np.argmin(cost_k_list) + 1)

    opt_k = elbow_choice(cost_k_list)

    print(
        f'-- Minimum cost is at k={min_cost_k}.'
    )

    print(
        f'-- Suggested elbow is at k={opt_k}.'
    )

    # ============= STEP 2: COST PLOT =============

    print('==> Step 2: Generating cost plot...')

    plt.style.use('ggplot')

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, 21),
        cost_k_list,
        'o-'
    )

    plt.axvline(
        opt_k,
        linestyle='--',
        color='gray',
        label=f'Elbow: k={opt_k}'
    )

    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Euclidean distances')

    plt.title('Cost vs Number of Clusters')

    plt.xticks(range(1, 21))
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        path.parent / 'kmeans_1.png',
        dpi=250
    )

    plt.close()

    # ============= STEP 3: VISUALIZATION =============

    print('==> Step 3: Visualizing clustering...')

    clusters, label = best_models[opt_k]

    plt.figure(figsize=(8, 6))

    # Plot data points colored by cluster
    data_plot = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=label.ravel(),
        cmap='tab20',
        s=8,
        alpha=0.65
    )

    # Mark cluster centers
    center_plot = plt.scatter(
        clusters[:, 0],
        clusters[:, 1],
        c='black',
        marker='X',
        s=150,
        edgecolors='white'
    )

    plt.legend(
        (data_plot, center_plot),
        ('data', 'clusters'),
        loc='best'
    )

    plt.title(
        f'Visualization with {opt_k} clusters'
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()

    plt.savefig(
        path.parent / 'kmeans_2.png',
        dpi=250
    )

    plt.close()

    print('==> Finished!')
    print('Saved kmeans_1.png and kmeans_2.png')