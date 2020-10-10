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
parser.add_argument("--rob")

args = parser.parse_args()
inference = str(args.opt)
rob = int(args.rob)

#inference = "VOGN"
#rob = 0
num_images = 50

model = PosteriorModel("%s_FCN_Posterior_%s"%(inference, rob))

epistemic, aleatoric = analyzers.variational_uncertainty(model, X_test[0:num_images])
auc = analyzers.auroc(model, X_test[0:num_images], y_test[0:num_images])

print("ON MNIST")
print("-------------------------------")
print("Epistemic: ",np.mean(epistemic))
print("Aleatoric: ",np.mean(aleatoric))
print("AUC: ", auc)
print("-------------------------------")

epistemic, aleatoric = analyzers.variational_uncertainty(model, F_train[0:num_images])
auc = analyzers.auroc(model, F_train[0:num_images], yf_train[0:num_images])

print("ON FasionMNIST")
print("-------------------------------")
print("Epistemic: ",np.mean(epistemic))
print("Aleatoric: ",np.mean(aleatoric))
print("AUC: ", auc)
print("-------------------------------")
llr = analyzers.likelihood_ratio(model, X_test[0:num_images], F_train[0:num_images])
print("LLR: ", llr)
print("-------------------------------")

meth = analyzers.FGSM
loss = tf.keras.losses.SparseCategoricalCrossentropy()
adv = meth(model, X_test[0:num_images], eps=0.1, loss_fn=loss, num_models=10, direction=y_test[0:num_images])

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
