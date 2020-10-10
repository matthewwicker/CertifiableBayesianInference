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

#inference = "VOGN"
#rob = 0
num_images = 100
for rob in range(3):
    model = PosteriorModel("%s_FCN_Posterior_%s"%(inference, rob))

    accuracy = tf.keras.metrics.Accuracy()
    preds = model.predict(X_test[0:500]) #np.argmax(model.predict(np.asarray(adv)), axis=1)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:500])
    fgsm = accuracy.result()
    print("%s Accuracy: "%(inference), accuracy.result())

    accuracy = tf.keras.metrics.Accuracy()
    adv = analyzers.FGSM(model, X_test[0:100], eps=0.1, loss_fn=loss, num_models=10)
    preds = model.predict(adv) #np.argmax(model.predict(np.asarray(adv)), axis=1)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:100])
    fgsm = accuracy.result()
    print("FGSM Robustness: ", accuracy.result())

    accuracy = tf.keras.metrics.Accuracy()
    preds = analyzers.chernoff_bound_verification(model, X_test[0:100], 0.1, y_test[0:100], confidence=0.80)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:100])
    print("Chernoff Lower Bound (IBP): ",  accuracy.result())

    epistemic, aleatoric = analyzers.variational_uncertainty(model, X_test[0:num_images])
    auc = analyzers.auroc(model, X_test[0:num_images], y_test[0:num_images])
    entropy = analyzers.predictive_entropy(model, X_test[0:num_images])
    print("ON MNIST")
    print("-------------------------------")
    print("Epistemic: ",np.mean(epistemic))
    print("Aleatoric: ",np.mean(aleatoric))
    print("AUC: ", auc)
    print("Entropy: ", np.mean(entropy))
    print("-------------------------------")
    m_ent = np.mean(entropy)

    epistemic, aleatoric = analyzers.variational_uncertainty(model, F_train[0:num_images])
    auc = analyzers.auroc(model, F_train[0:num_images], yf_train[0:num_images])
    entropy = analyzers.predictive_entropy(model, F_train[0:num_images])
    print("ON FasionMNIST")
    print("-------------------------------")
    print("Epistemic: ",np.mean(epistemic))
    print("Aleatoric: ",np.mean(aleatoric))
    print("AUC: ", auc)
    print("Entropy: ", np.mean(entropy))
    print("-------------------------------")
    llr = analyzers.likelihood_ratio(model, X_test[0:num_images], F_train[0:num_images])
    print("LLR %s: "%(rob), llr)
    print("EntR %s: "%(rob), np.mean(entropy)/m_ent)
    print("-------------------------------")
    meth = analyzers.FGSM
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    adv = meth(model, X_test[0:num_images], eps=0.3, loss_fn=loss, num_models=10, direction=y_test[0:num_images])

    epistemic, aleatoric = analyzers.variational_uncertainty(model, adv)
    auc = analyzers.auroc(model, adv, y_test[0:num_images])
    entropy = analyzers.predictive_entropy(model, adv)
    print("ON Adversarial Examples")
    print("-------------------------------")
    print("Epistemic: ",np.mean(epistemic))
    print("Aleatoric: ",np.mean(aleatoric))
    print("AUC: ", auc)
    print("Entropy: ", np.mean(entropy))
    print("-------------------------------")
    llr = analyzers.likelihood_ratio(model, X_test[0:num_images], adv)
    print("LLR %s: "%(rob), llr)
    print("EntR %s: "%(rob), np.mean(entropy)/m_ent)
    print("-------------------------------")
