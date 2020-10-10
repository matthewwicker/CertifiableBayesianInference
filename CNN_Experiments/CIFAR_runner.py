import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rob")
parser.add_argument("--eps")
parser.add_argument("--lam")
parser.add_argument("--opt")
parser.add_argument("--gpu", nargs='?', default='0,1,2,3,4,5')

args = parser.parse_args()
eps = float(args.eps)
lam = float(args.lam)
rob = int(args.rob)
optim = str(args.opt)
gpu = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import BayesKeras
import BayesKeras.optimizers as optimizers


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

augment_size = 40000
image_generator = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.075, 
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False, 
    data_format="channels_last",
    zca_whitening=False)
# fit data for zca whitening
image_generator.fit(X_train, augment=True)
# get transformed images
randidx = np.random.randint(50000, size=augment_size)
x_augmented = X_train[randidx].copy()
y_augmented = y_train[randidx].copy()
x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                            batch_size=augment_size, shuffle=False).next()[0]
# append augmented data to trainset
X_train = np.concatenate((X_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

#X_train = X_train[0:10000]
#y_train = y_train[0:10000]

model_type = "small"
if(model_type == "VGG8"):
    model = Sequential()
    #tf.keras.layers.GaussianNoise(stddev, **kwargs)
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='elu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='elu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='elu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))

elif(model_type == "small"):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(4, 4), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))

elif(model_type == "mini"):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))

elif(model_type == "medium"): 
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))

elif(model_type == "large"):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation = 'softmax'))

lr = 1
print("Got flag: %s"%(optim))
if(optim == 'VOGN'):
    learning_rate = 0.35*lr; decay=0.075
#    learning_rate = 1.5*lr; decay=0.0
    #learning_rate = 0.05*lr; decay=0.0
    opt = optimizers.VariationalOnlineGuassNewton()
elif(optim == 'BBB'):
    learning_rate = 0.5*lr; decay=0.0
    opt = optimizers.BayesByBackprop()
elif(optim == 'SWAG'):
    learning_rate = 0.05*lr; decay=0.125
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'NA'):
    learning_rate = 0.00075*lr; decay=0.075
    opt = optimizers.NoisyAdam()
elif(optim == 'SGD'):
    learning_rate = 0.05*lr; decay=0.1
    opt = optimizers.StochasticGradientDescent()
# Compile the model to train with Bayesian inference
if(rob == 0 or rob == 3 or rob == 4):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
else:
    loss = BayesKeras.optimizers.losses.robust_crossentropy_loss

inf = 2.5
#learning_rate *= 1.5

bayes_model = opt.compile(model, loss_fn=loss, epochs=25, learning_rate=learning_rate, batch_size=128, input_noise=0.0,
                          decay=decay, robust_train=rob, epsilon=eps, rob_lam=lam, inflate_prior=inf, log_path="%s_%s_Posterior_%s.log"%(optim, model_type, rob))

# Train the model on your data
bayes_model.train(X_train, y_train, X_test, y_test)

# Save your approxiate Bayesian posterior
bayes_model.save("%s_%s_Posterior_%s"%(optim, model_type, rob))
