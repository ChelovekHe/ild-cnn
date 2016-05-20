'''
this is a helper file to look at the MNIST original dataset as supplied by Marios
'''


import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils 
import sklearn.metrics as metrics


def load_data():
    # loading mnist dataset
    (X_train, y_train), (X_val, y_val) = mnist.load_data()

    # adding a singleton dimension and rescale to [0,1]
    X_train = np.asarray(np.expand_dims(X_train,1))/float(255)
    X_val = np.asarray(np.expand_dims(X_val,1))/float(255)

    # labels to categorical vectors
    uniquelbls = np.unique(y_train)
    nb_classes = uniquelbls.shape[0]
    zbn = np.min(uniquelbls) # zero based numbering
    y_train = np_utils.to_categorical(y_train-zbn, nb_classes)
    y_val = np_utils.to_categorical(y_val-zbn, nb_classes)

    return (X_train, y_train), (X_val, y_val)

# loading mnist data as example
(X_train, y_train), (X_val, y_val) = load_data()

X_train

