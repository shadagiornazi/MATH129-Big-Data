"""
MATH129: Big Data 
Homework 7, Problem 2
Lasso Feature Selection
Shada Giornazi 
Start file for hw7pr2 of Big Data Summer 2017

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

First, fill in the the code of step 0 in the main driver to load the data, then
please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============= HELPER FUNCTIONS =============

def find_cost(X, y, W, reg):
    """Calculate the L1-regularized squared error."""
    residual = X @ W - y
    cost = np.sum(residual ** 2) + reg * np.sum(np.abs(W[1:]))
    return float(cost)


def find_grad(X, y, W, reg=0.0):
    """Calculate the Lasso subgradient."""
    grad = 2 * X.T @ (X @ W - y)

    penalty = np.sign(W)
    penalty[0] = 0  # Do not regularize the intercept

    grad = grad + reg * penalty
    return grad


def prox(X, gamma):
    """Apply soft-thresholding to each entry of X."""
    X[:] = np.sign(X) * np.maximum(np.abs(X) - gamma, 0.0)
    return X



def grad_lasso(X, y, reg=1e6, lr=1e-12, eps=1e-5,
               max_iter=300, batch_size=256, print_freq=1):
    """Full-batch proximal gradient descent for Lasso."""

    m, n = X.shape
    obj_list = []

    # Initialize weights
    W = np.zeros((n, 1))

    # Lipschitz constant of squared-error gradient
    L = 2.0 * np.linalg.eigvalsh(X.T @ X)[-1]
    lr = 0.9 / L

    print('==> Running gradient descent...')
    t_start = time.time()

    for iteration in range(max_iter):

        # Full-batch gradient of squared error
        W_grad = 2.0 * X.T @ (X @ W - y)

        # Gradient step
        W_new = W - lr * W_grad

        # Soft-threshold coefficients (not intercept)
        prox(W_new[1:], lr * reg)

        # Calculate objective
        cost = find_cost(X, y, W_new, reg)
        obj_list.append(cost)

        change = np.linalg.norm(W_new - W)
        W = W_new

        if (iteration + 1) % print_freq == 0:
            sparsity = np.mean(np.isclose(W[1:], 0.0))

            print(
                f'-- Iteration {iteration + 1} '
                f'- training cost {cost:.4f} '
                f'- sparsity {sparsity:.2f}'
            )

        # Stop when weights stabilize
        if change < eps:
            break

    print(
        f'-- Time elapsed for training: '
        f'{time.time() - t_start:.2f} seconds'
    )

    return W, obj_list



def lasso_path(X, y, tau_min=1e-8,
               tau_max=10, num_reg=10):
    """Calculate weights for different regularization strengths."""

    m, n = X.shape

    W = np.zeros((n, num_reg))
    tau_list = np.linspace(tau_min, tau_max, num_reg)

    for index in range(num_reg):

        reg = 1.0 / tau_list[index]

        print(
            '-- Regularization parameter is {:.4E}'.format(reg)
        )

        # Run Lasso gradient descent
        W_fit, _ = grad_lasso(
    		X, y,
    		reg=reg,
    		max_iter=1000,
    		print_freq=1000
		)

        # Store optimal weights
        W[:, index] = W_fit.flatten()

    return W, tau_list


# ============= MAIN DRIVER =============

if __name__ == '__main__':

    # =============STEP 0: LOADING DATA=================
    print('==> Step 0: Loading data...')

    url = (
        'https://math189r.github.io/hw/data/'
        'online_news_popularity/online_news_popularity.csv'
    )

    try:
        df = pd.read_csv(
            url, sep=', ', engine='python'
        )

    except Exception as exc:
        print(f'Online download failed: {exc}')
        print('Trying local online_news_popularity.csv...')

        df = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')

    # Clean column names
    df.columns = df.columns.str.strip()

    feature_names = [
        col for col in df.columns
        if col not in ['url', 'shares', 'cohort']
    ]

    features = df[feature_names].apply(
        pd.to_numeric, errors='raise'
    )

    # Standardize predictors for stable Lasso optimization
    X_features = features.to_numpy(dtype=float)

    feature_means = X_features.mean(axis=0)
    feature_stds = X_features.std(axis=0)

    # Avoid division by zero for constant features
    feature_stds[feature_stds == 0] = 1.0

    X_features = (
        X_features - feature_means
    ) / feature_stds

    # Add intercept column
    X = np.hstack((
        np.ones((len(df), 1)),
        X_features
    ))


    y = np.log(
        df['shares'].to_numpy(dtype=float)
    ).reshape(-1, 1)

    print(
        f'Loaded {X.shape[0]} observations '
        f'and {len(feature_names)} features.'
    )

    # =============STEP 1: LASSO GRADIENT DESCENT=================
    # Implemented in helper functions above

    # =============STEP 2: LASSO PATH=================
    print('==> Step 2: Running lasso path...')

    W, tau_list = lasso_path(
    X, y,
    tau_min=1e-5,
    tau_max=2e-2,
    num_reg=30
	)

    plt.style.use('ggplot')

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10)
    )

    axes[0].plot(tau_list, W[1:, :].T)

    axes[0].set_title('Lasso Path')
    axes[0].set_xlabel(r'$\tau = \lambda^{-1}$')
    axes[0].set_ylabel(r'$W_i$')

    # =============STEP 3: FEATURE SELECTION=================
    print('==> Step 3: The most important features are:')

    # Use the final regularization path column
    final_weights = W[1:, -1]

    # Rank by absolute coefficient magnitude
    top_indices = np.argsort(
        np.abs(final_weights)
    )[::-1][:5]

    top_features = [
        (feature_names[i], float(final_weights[i]))
        for i in top_indices
    ]

    for rank, (name, weight) in enumerate(top_features, 1):
        print(
            f'{rank}. {name}: '
            f'coefficient = {weight:.6g}'
        )

    print('Feature rankings are based on standardized predictor coefficients.')

    # =============STEP 4: CONVERGENCE PLOT=================
    print('==> Step 4: Generating convergence plot...')

    W_reg, obj_list = grad_lasso(
        X, y,
        reg=1e5,
        lr=1e-12,
        eps=1e-2,
        max_iter=2500,
        batch_size=1024,
        print_freq=250
    )

    axes[1].plot(obj_list)

    axes[1].set_title(
        r'Lasso Objective Convergence: $\lambda = 10^5$'
    )

    axes[1].set_ylabel('Objective')
    axes[1].set_xlabel('Iteration')

    plt.tight_layout()

    plt.savefig(
        'hw7pr2_lasso.png',
        format='png',
        dpi=300
    )

    plt.close()

    print('==> Plotting completed.')
    print('Saved hw7pr2_lasso.png')