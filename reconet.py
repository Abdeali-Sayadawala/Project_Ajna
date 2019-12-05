# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import losses ,activations, optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Reshape, LeakyReLU, Dropout, Flatten, Lambda, BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop

def get_euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

class Reconet(object):
    
    def __init__(self):
        
        self.__DIM = 200
        
        seq_model_arch = [
                
            #Reshape(input_shape=((self.__DIM**2)*3 , ), target_shape=(self.__DIM, self.__DIM, 3)),
                      
            Conv2D(32, kernel_size=(2,2), activation='relu'),
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            
            Conv2D(64, kernel_size=(1,1), activation='relu'),
            MaxPooling2D(pool_size=(2,2), strides=2),
            
            Conv2D(32, kernel_size=(2,2), activation='relu'),
            Conv2D(64, kernel_size=(1,1), activation='relu'),
            MaxPooling2D(pool_size=(4,4), strides=2),            
            Dropout(0.15),
                        
            Flatten(),
            
            Dense(128, activation = 'relu'),
            Dropout(0.1),
            Dense(50, activation = 'relu')   
        ]
        
        seq_model = Sequential(seq_model_arch)
        
        input1 = Input(shape=(self.__DIM, self.__DIM, 3))
        input2 = Input(shape=(self.__DIM, self.__DIM, 3))
        
        output1 = seq_model(input1)
        output2 = seq_model(input2)
        
        distance = Lambda(get_euclidean_distance, output_shape=eucl_output_shape)([output1, output2])
                
        rms = RMSprop()
        self.__model = Model([input1, input2], distance)
        
        def contrastive_loss(y_true, y_pred):
            margin = 1
            return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
        
        self.__model.compile(loss = contrastive_loss, optimizer=rms)
        
        rms = RMSprop()
        
        #for debugging and developing. Commnet while implementing or unnecessary printing in console
        #print(self.__model.summary())
        
    def fit(self, x, y, hyperparameters):
        self.__model.fit(x, y, 
                         batch_size = hyperparameters['batch_size'],
                         epochs = hyperparameters['epochs'],
                         callbacks = hyperparameters['callbacks'],
                         validation_data = hyperparameters['val_data']
        )
        
    def predict(self, x):
        return self.__model.predict(x)
    
    def prepare_images(self, path, flatten = False):
        images = list()
        for img in os.listdir(path):
            image = cv2.imread(str(path)+"/"+str(img))
            image = cv2.resize(image, (self.__DIM, self.__DIM))
            image = np.array(np.reshape(image, (self.__DIM, self.__DIM, 3)))/255
            images.append(image)
        
        if flatten == True:
            images = np.array(images)
            return images.reshape((images.shape[0], self.__DIM**2 * 3)).astype(np.float32)
        else:
            return np.array(images)
        
    def save_model(self, path):
        self.__model.save(path)
        
    def load_model(self, path):
        self.__model = load_model(path)
    
    
rec = Reconet()