# Matthew Wicker
import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))


import BayesKeras
from BayesKeras import PosteriorModel
from BayesKeras import analyzers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np


(F_train, yf_train), (Xf_test, yf_test) = tf.keras.datasets.fashion_mnist.load_data()
F_train = F_train/255.
F_train = F_train.astype("float32").reshape(-1, 28*28)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opt")

args = parser.parse_args()
inference = str(args.opt)

loss = tf.keras.losses.SparseCategoricalCrossentropy()

num_images = 500
for rob in range(3):
    model = PosteriorModel("%s_FCN_Posterior_%s"%(inference, rob))
    entropy = analyzers.predictive_entropy(model, X_test[0:num_images])
    print(entropy)
    print("========================== %s =========================="%(rob))
    adv = analyzers.PGD(model, X_test[0:num_images], eps=0.75, loss_fn=loss, num_models=35)
    accuracy = tf.keras.metrics.Accuracy()
    preds = model.predict(adv) #np.argmax(model.predict(np.asarray(adv)), axis=1)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
    print("FGSM Robustness: ", accuracy.result())
    fentropy = analyzers.predictive_entropy(model, adv)
#    fentropy = analyzers.predictive_entropy(model, F_train[0:num_images])
    print(fentropy)
