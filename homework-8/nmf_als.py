"""
MATH 129: Big Data 
Homework 8
Shada Giornazi 
Extra Credit: Non-Negative Matrix Factorization
Reuters Topic Modeling

Starter file for nmf of Big Data Summer 2017

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
	3) Let k be the number of topics

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction import text
from pathlib import Path


# ============= HELPER FUNCTIONS =============

def nmf_cost(X, W, H):
    """Calculate squared Frobenius reconstruction error."""

    # ||X-WH||^2 = ||X||^2 - 2<X,WH> + ||WH||^2
    x_norm = X.multiply(X).sum()

    XH = X @ H.T
    cross_term = np.sum(W * XH)

    WtW = W.T @ W
    HHt = H @ H.T

    reconstruction_norm = np.sum(WtW * HHt)

    cost = x_norm - 2.0 * cross_term + reconstruction_norm

    return float(max(cost, 0.0))


def nmf(X, k=20, max_iter=100, print_freq=5):
    """NMF using Lee-Seung multiplicative updates."""

    m, n = X.shape
    rng = np.random.default_rng(2026)

    W = rng.random((m, k)) * 1e-2
    H = rng.random((k, n)) * 1e-2

    eps = 1e-12
    cost_list = [nmf_cost(X, W, H)]

    t_start = time.time()

    for iter_num in range(max_iter):

        # Update H while holding W fixed
        numerator_H = W.T @ X
        denominator_H = (W.T @ W) @ H

        H *= numerator_H / np.maximum(denominator_H, eps)

        # Update W while holding H fixed
        numerator_W = X @ H.T
        denominator_W = W @ (H @ H.T)

        W *= numerator_W / np.maximum(denominator_W, eps)

        # Calculate objective
        cost = nmf_cost(X, W, H)
        cost_list.append(cost)

        if (iter_num + 1) % print_freq == 0:
            print(
                f'-- Iteration {iter_num + 1} '
                f'- cost: {cost:.4E}'
            )

    t_end = time.time()

    print(
        '-- Time elapsed for running nmf: '
        f'{t_end - t_start:.2f} seconds'
    )

    return W, H, cost_list


# ============= MAIN DRIVER =============

if __name__ == '__main__':

    # ============= STEP 0: LOADING DATA =============

    print('==> Step 0: Loading Reuters data...')

    try:
        reuters.fileids()
    except LookupError:
        nltk.download('reuters', raise_on_error=True)

    documents = [
        ' '.join(reuters.words(file_id)).lower()
        for file_id in reuters.fileids()
    ]

    tfidf = text.TfidfVectorizer(
        stop_words='english',
        max_features=10000
    )

    X = tfidf.fit_transform(documents)

    print(
        f'Loaded {X.shape[0]} documents '
        f'and {X.shape[1]} features.'
    )

    # ============= STEP 1: RUNNING NMF =============

    print('==> Step 1: Running NMF...')

    num_topics = 20

    W, H, cost_list = nmf(
        X,
        k=num_topics,
        max_iter=100,
        print_freq=10
    )

    # ============= STEP 2: CONVERGENCE CHECK =============

    print('==> Step 2: Checking convergence...')

    differences = np.diff(cost_list)

    tolerance = 1e-7 * max(1.0, cost_list[0])

    is_nonincreasing = np.all(differences <= tolerance)

    is_nonnegative = (
        np.all(W >= 0) and np.all(H >= 0)
    )

    print(f'Initial cost: {cost_list[0]:.6f}')
    print(f'Final cost: {cost_list[-1]:.6f}')

    print(
        'Objective non-increasing: '
        f'{is_nonincreasing}'
    )

    print(
        'Factors nonnegative: '
        f'{is_nonnegative}'
    )

    print(
        'Maximum cost increase: '
        f'{max(0.0, np.max(differences)):.6e}'
    )

    # ============= STEP 3: CONVERGENCE PLOT =============

    print('==> Step 3: Generating convergence plot...')

    plt.style.use('ggplot')
    plt.figure(figsize=(9, 6))

    plt.plot(
        range(len(cost_list)),
        cost_list,
        linewidth=2
    )

    plt.xlabel('Iteration')
    plt.ylabel('Squared Frobenius Reconstruction Error')

    plt.title(
        f'Reuters NMF Convergence with {num_topics} Topics'
    )

    plt.tight_layout()
    plt.savefig('nmf_cvg.png', dpi=250)
    plt.close()

    # ============= STEP 4: TOP WORDS =============

    print('==> Step 4: Finding top words for each topic...')

    num_top_words = 20

    ind = np.argsort(H, axis=1)[:, -num_top_words:]
    ind = np.flip(ind, axis=1)

    feature_names = np.array(
        tfidf.get_feature_names_out()
    )

    top_words = feature_names[ind]

    output_lines = []

    for topic_ind in range(H.shape[0]):

        words = ', '.join(top_words[topic_ind])

        line = f'Topic {topic_ind + 1}: {words}'

        print(line)
        output_lines.append(line)

    Path('nmf_topics.txt').write_text(
        '\n'.join(output_lines),
        encoding='utf-8'
    )

    print('==> Finished!')
    print('Saved nmf_cvg.png and nmf_topics.txt')