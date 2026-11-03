"""
MATH129: Big Data 
Homework 6 
Shada Giornazi 
Starter file for hw6pr2 of Big Data Summer 2017

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
	1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import urllib.request


if __name__ == '__main__':

    # =============STEP 0: LOADING DATA=================
    print('==> Loading image data...')

    image_url = 'https://i.imgur.com/X017qGH.jpg'

    try:
        with urllib.request.urlopen(image_url, timeout=20) as response:
            img = np.array(
                Image.open(BytesIO(response.read())).convert('L'),
                dtype=float
            )
    except Exception:
        print('Could not download image. Loading local clown.jpg...')
        img = np.array(Image.open('clown.jpg').convert('L'), dtype=float)

    # Shuffle the image
    shuffle_img = img.flatten().copy()
    np.random.shuffle(shuffle_img)

    # Reshape the shuffled image
    shuffle_img = shuffle_img.reshape(img.shape)

    # =============STEP 1: RUNNING SVD ON IMAGES=================
    print('==> Running SVD on images...')

    # SVD on original image
    U, S, V = np.linalg.svd(img, full_matrices=False)

    # SVD on shuffled image
    U_s, S_s, V_s = np.linalg.svd(
        shuffle_img, full_matrices=False
    )

    # =============STEP 2: SINGULAR VALUE DROPOFF=================
    print('==> Singular value dropoff plot...')

    k = min(100, len(S), len(S_s))

    plt.style.use('ggplot')

    orig_S_plot, = plt.plot(
        range(1, k + 1),
        S[:k],
        color='blue',
        label='Original'
    )

    shuf_S_plot, = plt.plot(
        range(1, k + 1),
        S_s[:k],
        color='red',
        label='Shuffled'
    )

    plt.legend(
        (orig_S_plot, shuf_S_plot),
        ('original', 'shuffled'),
        loc='best'
    )

    plt.title('Singular Value Dropoff for Clown Image')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Values')

    plt.tight_layout()
    plt.savefig('dropoff.png', format='png', dpi=300)
    plt.close()

    # =============STEP 3: RECONSTRUCTION=================
    print('==> Reconstruction with different ranks...')

    rank_list = [2, 10, 20]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='Greys_r', vmin=0, vmax=255)
    plt.axis('off')
    plt.title('Original Image')

    # Generate reconstruction images for each rank
    for index in range(len(rank_list)):

        k = rank_list[index]

        plt.subplot(2, 2, 2 + index)

        # Rank-k truncated SVD approximation
        reconstruction = (
            U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
        )

        plt.imshow(
            reconstruction,
            cmap='Greys_r',
            vmin=0,
            vmax=255
        )

        plt.title('Rank {} Approximation'.format(k))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction.png', format='png', dpi=300)
    plt.close()

    print('==> Finished!')
    print('Saved dropoff.png and reconstruction.png')