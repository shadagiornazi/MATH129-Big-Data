
"""
Used to load and process the MNIST data for Problem 2.
"""

print("==>Loading data...")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


# Load MNIST
mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False
)

# Get image data and labels
X = mnist.data.astype(float)
y = mnist.target.astype(int).reshape(-1, 1)


# Use standard MNIST training/testing split
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]


# Create column names expected by the homework
name_list = ['label']

for i in range(784):
    name_list.append('pix_' + str(i))


# Create data frames in the same format as the original starter
df_train = pd.DataFrame(
    np.hstack((y_train, X_train)),
    columns=name_list
)

df_test = pd.DataFrame(
    np.hstack((y_test, X_test)),
    columns=name_list
)


# Make labels integers
df_train['label'] = df_train['label'].astype(int)
df_test['label'] = df_test['label'].astype(int)


# Scale data for Part B
X_train = X_train / 256.
X_test = X_test / 256.


print("==>Data loaded successfully.")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
