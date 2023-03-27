# Neural Network from numpy

import numpy as np
from sklearn import datasets as ds

# get inputs
iris_X, iris_y = ds.load_iris(return_X_y=True)
train_split = round(0.8*iris_X.shape[0])
train_inputs = iris_X[0:train_split]
train_targets = iris_y[0:train_split]

test_inputs = iris_X[train_split:]
test_targets = iris_y[train_split:]


# create layer(s)

# train
    # -move forward through layers (math)
    # -generate error signal
    # - backpropogate and update weights and bias
    # - loop

# test
