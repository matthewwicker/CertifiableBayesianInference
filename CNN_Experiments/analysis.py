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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opt")
parser.add_argument("--rob")

args = parser.parse_args()
opt = str(args.opt)
rob = int(args.rob)

inference = opt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

model_type = "small"

model = PosteriorModel("%s_%s_Posterior_%s"%(inference, model_type,  rob))
from tqdm import trange

loss = tf.keras.losses.SparseCategoricalCrossentropy()


#print(model._predict(X_test[0:1]))
#print(analyzers.IBP(model, X_test[0:1], eps=0.0, weights=model.model.get_weights(), predict=True))
accuracy = tf.keras.metrics.Accuracy()
preds = model.predict( X_test[0:250])
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:250])
print("Accuracy: ",  accuracy.result())
accuracy = tf.keras.metrics.Accuracy()
adv = analyzers.FGSM(model, X_test[0:250], eps=0.0035, loss_fn=loss, num_models=5)
preds = model.predict(adv) #np.argmax(model.predict(np.asarray(adv)), axis=1)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:250])
fgsm = accuracy.result()
print("FGSM Robustness: ", accuracy.result())

accuracy = tf.keras.metrics.Accuracy()
preds = analyzers.chernoff_bound_verification(model, X_test[0:250], 0.0035, y_test[0:250], confidence=0.90)
#print(preds.shape)
#print(np.argmax(preds, axis=1).shape)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:250])
print("Massart Lower Bound (IBP): ",  accuracy.result())

