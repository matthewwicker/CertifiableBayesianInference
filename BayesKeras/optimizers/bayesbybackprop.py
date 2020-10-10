#Author: Matthew Wicker
# Impliments the BayesByBackprop optimizer for BayesKeras

import os
import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tqdm import tqdm
from tqdm import trange

from BayesKeras.optimizers import optimizer 
from BayesKeras.optimizers import losses
from BayesKeras import analyzers
from abc import ABC, abstractmethod

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class BayesByBackprop(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.15, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)


        # Now we get into the BayesByBackprop specific enrichments to the class
       
        # Post process our variances to all be stds:
        for i in range(len(self.posterior_var)):
            self.posterior_var[i] = tf.math.log(tf.math.exp(self.posterior_var[i])-1)
        self.kl_weight = kwargs.get('kl_weight', 1.0)              
        self.kl_component = tf.keras.metrics.Mean(name="kl_comp")  
        print("BayesKeras: Using passed loss_fn as the data likelihood in the KL loss")
        
        return self

    def step(self, features, labels, lrate):
        """
        Initial sampling for BBB
        """
        init_weights = []; noise_used = []
        for i in range(len(self.posterior_mean)):
            noise = tf.random.normal(shape=self.posterior_var[i].shape, 
                                     mean=tf.zeros(self.posterior_var[i].shape), stddev=1.0)
            var_add = tf.multiply(softplus(self.posterior_var[i]), noise)
            #var_add = tf.multiply(self.posterior_mean[i], noise)
            w = tf.math.add(self.posterior_mean[i], var_add)
            noise_used.append(noise)
            init_weights.append(w)
        self.model.set_weights(init_weights)

        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:   # Below we add an extra variable for IBP
            tape.watch(self.posterior_mean) 
            tape.watch(self.posterior_var); #tape.watch(init_weights)
            if(self.robust_train == 0):
                predictions = self.model(features)
                worst_case = predictions # cheap hack lol
                loss, kl_comp = losses.KL_Loss(labels, predictions, self.model.trainable_variables,
                                               self.prior_mean, self.prior_var, 
                                               self.posterior_mean, self.posterior_var, 
                                               self.loss_func, self.kl_weight)
            elif(int(self.robust_train) == 1):
                # Get the probabilities
                predictions = self.model(features)
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                #!*! TODO: Undo the hardcoding of depth in this function
                v1 = tf.one_hot(labels, depth=10)
                v2 = 1 - tf.one_hot(labels, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))

                # Now we have the worst case softmax probabilities
                worst_case = self.model.layers[-1].activation(worst_case)
                # Calculate the loss
                loss, kl_comp = losses.robust_KL_Loss(labels, predictions, self.model.trainable_variables,
	       		                              self.prior_mean, self.prior_var, 
                                                      self.posterior_mean, self.posterior_var, 
                                                      self.loss_func, self.kl_weight, 
                                                      worst_case, self.robust_lambda)
            
            elif(int(self.robust_train) == 2):
                predictions = self.model(features)
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                # Get the probabilities
                worst_case = self.model(features_adv)
                #print(predictions[0], worst_case[0])
                # Calculate the loss
                loss, kl_comp = losses.robust_KL_Loss(labels, predictions, self.model.trainable_variables,
                                                      self.prior_mean, self.prior_var,
                                                      self.posterior_mean, self.posterior_var,
                                                      self.loss_func, self.kl_weight,
                                                      worst_case, self.robust_lambda)
            # !*! Now this is the broken branch
            elif(int(self.robust_train) == 3):
                sys.exit(0)
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                v1 = tf.one_hot(labels, depth=10)
                v2 = 1 - tf.one_hot(labels, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                worst_case = self.model.layers[-1].activation(worst_case)
                one_hot_cls = tf.one_hot(labels, depth=10)
                output = tf.math.reduce_max((self.robust_lambda*(predictions*one_hot_cls))  + ((1-self.robust_lambda)*(worst_case*one_hot_cls)), axis=1)
                loss = self.loss_func(labels, predictions)
        # Get the gradients
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
        mean_gradient = tape.gradient(loss, self.posterior_mean)
        var_gradient = tape.gradient(loss, self.posterior_var)
        #init_gradient = tape.gradient(loss, init_weights)
        
        posti_mean_grad = []
        posti_var_grad = []
        # !*! - Make the weight and init gradients the same variable and retest
        for i in range(len(mean_gradient)):
            #weight_gradient[i] = tf.math.add(weight_gradient[i], init_gradient[i])
            weight_gradient[i] = tf.cast(weight_gradient[i], 'float32')
            mean_gradient[i] = tf.cast(mean_gradient[i], 'float32')
            f = tf.math.add(weight_gradient[i], mean_gradient[i])
            posti_mean_grad.append(f)
            v = tf.math.divide(noise_used[i], 1+tf.math.exp(tf.math.multiply(self.posterior_var[i], -1)))
            v = tf.math.multiply(v, weight_gradient[i])
            v = tf.math.add(v, var_gradient[i])
            posti_var_grad.append(v)
        #gradients = posti_mean_grad

        # APPLICATION OF WEIGHTS
        new_posti_var = []; new_posti_mean = []
        for i in range(len(mean_gradient)):
            pdv = tf.math.multiply(posti_var_grad[i], lrate)
            pdm = tf.math.multiply(posti_mean_grad[i], lrate)
            v = tf.math.subtract(self.posterior_var[i], pdv)
            m = tf.math.subtract(self.posterior_mean[i], pdm)
            new_posti_var.append(v)
            new_posti_mean.append(m)

        self.train_loss(loss)
        self.train_metric(labels, predictions)
        #self.train_rob(labels, worst_case)
        self.kl_component(kl_comp)
        self.posterior_mean = new_posti_mean
        self.posterior_var = new_posti_var
        return new_posti_mean, new_posti_var

    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        super().train(X_train, y_train, X_test, y_test)

    def sample(self):
        sampled_weights = []
        for i in range(len(self.posterior_mean)):
            sampled_weights.append(np.random.normal(loc=self.posterior_mean[i],
                                                    scale=softplus(self.posterior_var[i])))
        return sampled_weights


    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        var = []
        for i in range(len(self.posterior_var)):
            var.append(softplus(self.posterior_var[i]))
        np.save(path+"/mean", np.asarray(self.posterior_mean))
        np.save(path+"/var", np.asarray(var))
        self.model.save(path+'/model.h5')
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)


    
