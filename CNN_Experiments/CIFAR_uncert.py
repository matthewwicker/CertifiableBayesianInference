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

def svhn_load():
    X_test = np.load("../SVHN/data/svhn/cropped/cropped_test_imgs.npy")
    y_test = np.load("../SVHN/data/svhn/cropped/cropped_test_labels.npy")
    X_train = np.load("../SVHN/data/svhn/cropped/cropped_train_imgs.npy")
    y_train = np.load("../SVHN/data/svhn/cropped/cropped_train_labels.npy")
    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)
    return (X_train, y_train), (X_test, y_test)

(F_train, yf_train), (Xf_test, yf_test) = svhn_load()
F_train = F_train/255.
F_train = F_train.astype("float32")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

inference = "NA"
rob = 0
num_images = 250

model = PosteriorModel("%s_FCN_Posterior_%s"%(inference, rob))
epistemic, aleatoric = analyzers.variational_uncertainty(model, X_test[0:num_images])
auc = analyzers.auroc(model, X_test[0:num_images], y_test[0:num_images])

print("ON CIFAR")
print("-------------------------------")
print("Epistemic: ",np.mean(epistemic))
print("Aleatoric: ",np.mean(aleatoric))
print("AUC: ", auc)
print("-------------------------------")

epistemic, aleatoric = analyzers.variational_uncertainty(model, F_train[0:num_images])
auc = analyzers.auroc(model, F_train[0:num_images], yf_train[0:num_images])

print("ON SVHN")
print("-------------------------------")
print("Epistemic: ",np.mean(epistemic))
print("Aleatoric: ",np.mean(aleatoric))
print("AUC: ", auc)
print("-------------------------------")

llr = analyzers.likelihood_ratio(model, X_test[0:100], F_train[0:100])
print("LLR: ", llr)

meth = analyzers.FGSM
loss = tf.keras.losses.SparseCategoricalCrossentropy()
adv = meth(model, X_test[0:num_images], eps=0.01, loss_fn=loss, num_models=10, direction=y_test[0:num_images])

epistemic, aleatoric = analyzers.variational_uncertainty(model, adv)
auc = analyzers.auroc(model, adv, y_test[0:num_images])
print("ON Adversarial Examples")
print("-------------------------------")
print("Epistemic: ",np.mean(epistemic))
print("Aleatoric: ",np.mean(aleatoric))
print("AUC: ", auc)
print("-------------------------------")
llr = analyzers.likelihood_ratio(model, X_test[0:num_images], adv)
print("LLR: ", llr)
print("-------------------------------")
