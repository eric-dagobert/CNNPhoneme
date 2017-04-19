#!/usr/bin/python -O
#from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
import time
from collections import defaultdict
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
#import theano
#import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,Flatten,TimeDistributed,Reshape,BatchNormalization,Bidirectional,MaxPooling3D
from keras.layers import Embedding,Input
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,LSTM,Conv2D,MaxPooling2D,SeparableConv2D,ConvLSTM2D,Conv3D
from keras.callbacks import LambdaCallback
from keras.layers.advanced_activations import LeakyReLU 

def train_eval():
    data = np.load('spec_6436mod.npy')
    labels = np.load('spec_6436mod_vec.npy').astype(int)
    w = data.shape[2]
    h = data.shape[1]      
    #data  = keras.utils.normalize(data)
    
    #labels = norm_labels(labels)
    SUBS  = -1
    OUTFEAT=labels.max()+1
    keras_nn5d(data, labels,h ,w, OUTFEAT,50,SUBS)
   
def save_set(x_t,x_e,y_t,y_e):
    np.save('x_t36a', x_t)
    np.save('x_e36a', x_e)
    np.save('y_t36', y_t)            
    np.save('y_e36', y_e)
    
def load_set():
    return np.load('x_t36.npy'),np.load('x_e36.npy'),np.load('y_t36.npy'),np.load('y_e36.npy') 
    
    
def keras_nn5d(X,labels,h,w,d,epochs,subs=-1,replay=True):   
    model = None
    def pp(b,logs):
        print 'batch#',logs['batch']
        #if logs['batch'] == 0:    
         #   model.save('model_tmp.kn')


    blabels = keras.utils.to_categorical(labels, num_classes=d).reshape(-1,4,d)
    
    X = X.reshape(-1,h,w,1)
    if subs > -1:
        X,_,blabels,_ =   train_test_split(X, blabels, test_size=1-subs, random_state=42)      
    if replay:
        x_train, x_eval, y_train, y_eval = load_set()
    else:
        x_train, x_eval, y_train, y_eval = train_test_split(X, blabels, test_size=0.2, random_state=42)       
        save_set(x_train, x_eval, y_train, y_eval )
        
    batch_print_callback =keras.callbacks.LambdaCallback(on_batch_begin=pp) 
    if replay:
        model = keras.models.load_model('model_convect5d36.kn')
    else:
        model= rnn(h, w, d,True)
        #model = keras.models.load_model('model_reducedconv.kn')
        model.compile(#loss = keras.losses.mean_squared_error  ,
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'],
                      callbacks=[batch_print_callback],
                      optimizer='adam')
                      #optimizer = 'sgd')
        
    for i in range(0,10):
        model.fit(x_train, y_train, batch_size=1000, epochs=epochs,callbacks=[batch_print_callback],shuffle=True)
        model.save('model_convect5d36.kn') 
        score = model.evaluate(x_eval, y_eval, batch_size=x_eval.shape[0]) 
        print
        print score

def rnn(h,w,d,multid = False):
    model = Sequential()
    model.add(Conv2D(64, (6, 6),input_shape=(h,w,1)))        
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3))) 
    #model.add(Conv2D(64, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(Conv2D(64, (6, 6)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(4, 4)))
    #model.add(Conv2D(64, (2, 2)))
    #model.add(Activation('relu'))   
        
    #model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(0.3))
       
    model.add(Flatten())
    #model.add(Dense(h*w))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))    
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    if multid:
        model.add(Dense(4*d))
        model.add(Reshape((4,d)))
    else:
        model.add(Dense(d))
    model.add(Activation('softmax'))
    #model.add(Activation('sigmoid'))
    #model.add(Activation('relu'))
    #model.add(LeakyReLU())
    #model.add(Reshape((5,d)))
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')       
    return model


train_eval()
