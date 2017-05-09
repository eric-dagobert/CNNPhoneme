#from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os

import time
from collections import defaultdict
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#import theano
#import theano.tensor as T
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers.recurrent import LSTM,SimpleRNN,GRU
from keras.layers.merge import Concatenate
from keras.layers import Dense, Dropout,Activation,Flatten,TimeDistributed,Reshape,BatchNormalization,Bidirectional,MaxPooling3D,GlobalAveragePooling2D
from keras.layers import Embedding,Input
from keras.layers import Convolution2D,MaxPooling2D,ZeroPadding2D,Conv2D
from keras.callbacks import LambdaCallback
from keras.layers.advanced_activations import LeakyReLU , ELU
import featuredic

def rebalance(data,y,size):
    labels = y
    hashed = np.array([str(tuple(x)) for x in labels[:,]])
    dic = defaultdict()  
    counts = [hashed[hashed == c].shape[0] for c in np.unique(hashed)]
    cm = max(counts)
    
    ct = size//np.unique(hashed).shape[0]
    datab , yb = None,None
    for c in np.unique(hashed):
        sup = np.random.choice(np.where(hashed==c)[0],size=ct)
        datab = data[sup] if datab is None else np.vstack((datab,data[sup]))
        yb =y[sup] if yb is None else np.vstack((yb,y[sup]))
    return datab,yb

def rnnv1(h,w,d,multid = False):
    model = Sequential()
    model.add(Conv2D(128, (6, 6),input_shape=(h,w,1)))        
    print 'output shape', model.output_shape     
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4))) 
    model.add(Dropout(0.3))    
   
    print 'output shape', model.output_shape     
    sh =model.output_shape
    print 'output size',sh[1]*sh[2]*sh[3]	
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dropout(.1))
    print 'output shape', model.output_shape     
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Activation('elu')) 
    model.add(Dense(2048))
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Dropout(.2))
    model.add(Activation('elu'))
    if multid:
        model.add(Dense(5*d))
        model.add(Reshape((5,d)))
        model.add(Activation('elu'))
        model.add( TimeDistributed(Dense(d),input_shape=(5,d)))
    else:
        model.add(Dense(d))
    model.add(Activation('softmax'))
    #model.add(Activation('sigmoid'))
    return model
    

    
def rnn16 (h,w,d):
    model = Sequential()
    #model.add(ZeroPadding2D((1,1),input_shape=(h,w,3)))
    #model.add(Convolution2D(128, 8,8,  activation='relu',input_shape=(h,w,3)))
    model.add(Conv2D(128, (8,8), input_shape=(h,w,3),activation='sigmoid'))
    #model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.4))
    
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='sigmoid'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2), strides=(2,2)))    
        
    #model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='sigmoid'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))    

    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(256, (3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(256, (3, 3), activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    
    #model.add(Conv2D(256, 3, 3, activation='relu'))
    """"model.add(MaxPooling2D((2,2), strides=(2,2)))    
    model.add(ZeroPadding2D((1,1)))
     
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    """
    model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(units=4*d,activation='relu'))
    
    model.add(Reshape((4,256)))
    model.add(LSTM(256,return_sequences=True,activation='sigmoid'))
    model.add(Dense(d,activation='linear'))
    
    #model.add(Dense(units=4*d))
    #model.add(Activation('softmax'))
    return model

def rnn(h,w,d,multid = True):
    model = Sequential()
    model.add(Conv2D(128, (8, 8),input_shape=(w,h,3)))        
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3))) 
    model.add(Dropout(0.25))
       
    model.add(Flatten())
   
    
    #model.add(Dense(h*w))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))    
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    if multid:
        model.add(Dense(units=4*d))
        model.add(Reshape((4,d)))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(output_dim=d,activation='softmax'))    
    return model

def rnn2(h,w,d,multid = False):
    model = Sequential()
    #model.add(Conv2D(64, (6, 6),input_shape=(h,w,1)))        
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))     
    
    #model.add(Conv2D(128, (8, 8),input_shape=(h,w,1)))        
    model.add(Conv2D(128, (8, 8),input_shape=(h,w,1)))        
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    #model.add(Dropout(0.4))    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))    
    #model.add(BatchNormalization())
    #model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('elu'))
    #model.add(Activation('relu'))
    model.add(Dense(2048))
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Activation('elu'))
    model.add(Dense(2048))
    model.add(Activation('elu'))
    #model.add(Dense(512))
    #model.add(Activation('elu'))
    model.add(Reshape((4,512)))
    #model.add(LSTM(4,return_sequences=True,activation='relu'))

    model.add(Bidirectional(LSTM(4,return_sequences=True,activation='relu')))
   
    #model.add(Dense(4*d))
    #model.add(Reshape((4,d)))
    model.add(Dense(d))
    model.add(Activation('softmax'))
    
 
   
    return model



