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
from keras.optimizers import SGD
import libnet
import featuredic
import time
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,libn.cnmem=1"
POSTFIX='extphtd.npy'
MODEL='model_cv2td'
DATA= 'spec_6440'
REPLAY=True
EPOCHS=10
CYCLESIZE=3000
MAX=4*12
def train_eval():
    data = np.load(DATA + '.npy')
    labels = np.load(DATA  + 'vec.npy').astype(int)
    w = data.shape[2]
    h = data.shape[1]      
    #data  = keras.utils.normalize(data)
    
    #labels = norm_labels(labels)
    SUBS  = -1
    OUTFEAT=labels.max()+1
    keras_nn5d(data, labels,h ,w, OUTFEAT,EPOCHS,REPLAY)
    
def save_set(x_t,x_e,y_t,y_e):
    np.save('x_t'+POSTFIX, x_t)
    np.save('x_e'+ POSTFIX, x_e)
    np.save('y_t'+POSTFIX, y_t)            
    np.save('y_e'+POSTFIX, y_e)
    
def load_set():
    return np.load('x_t'+POSTFIX),np.load('x_e'+POSTFIX),np.load('y_t'+POSTFIX),np.load('y_e'+POSTFIX)    


def keras_nn5d(X, labels,h ,w, d,epochs,replay=True):
    model = None
    def pp(b,logs):
	print 'batch#',logs['batch']
    X = X.reshape(-1,h,w,1)
    if replay:
        x_ub, x_eval, y_ub, y_eval = load_set()
    else:
        x_ub, x_eval, y_ub, y_eval = train_test_split(X, labels, test_size=0.2, random_state=46)    
        save_set(x_ub, x_eval, y_ub, y_eval )
        
    model= libnet.rnnv1(h, w, d,True)
    sgd=SGD(lr=.005,decay=1e-6,momentum=0.9,nesterov=True)
    
    if replay:
	model.load_weights(MODEL+'.we')

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'],
        optimizer = 'adadelta')
    bscore = 0
    for i in range(0,MAX):
	st = time.time()
	print 'cycle',i
	#for ph in featuredic._phonetov.keys():
	#    wt =  featuredic.featwhere(y_ub, ph)
	#   we = featuredic.featwhere(y_eval, ph)
	# print 'training of ',ph
	#train_loop(x_ub[wt],y_ub[wt],x_eval[we],y_eval[we],model,d,1.,phonemode=True)
	print 'training all'
	bscore = train_loop(x_ub,y_ub,x_eval,y_eval,model,d,bscore,phonemode=False)
	print 'cycle time : ', (time.time()-st)/60.,'mn'
	
    
def tobin(y,d,multid=True):
    if multid:
	return keras.utils.to_categorical(y, num_classes=d).reshape(-1,5,d)
    else:
	yb = np.zeros((y.shape[0],d))
	for i,vec in enumerate(y[:,]):
	    for x in vec:	
		vb = keras.utils.to_categorical(x, num_classes=d).reshape((d,))
		yb[i,] += vb
	return yb
    
def train_loop(x_ub,y_ub,x_eval,y_eval,model,d,bscore,phonemode=False):
    pscore=0    
    np.random.seed()
    if phonemode:
	x_train,y_train = libnet.rebalance(x_ub,y_ub,CYCLESIZE)   
    else:
	x_train,y_train = libnet.rebalance(x_ub,y_ub,CYCLESIZE)
    y_trainb = tobin(y_train,d)
    #y_trainb = keras.utils.to_categorical(y_train,
    #num_classes=d).reshape(-1,5,d)
    
    print 'xtrain size',x_train.shape
    model.fit(x_train, y_trainb, batch_size=1000, epochs=EPOCHS,shuffle=True)
    #x_evalb,y_evalb = x_eval[rde],y_eval[rde] 
    if phonemode:
	x_evalb,y_evalb = x_eval,y_eval
    else:
	x_evalb,y_evalb = libnet.rebalance(x_eval, y_eval, CYCLESIZE//5)
    y_evalb = tobin(y_evalb,d)

    score = model.evaluate(x_evalb, y_evalb, batch_size=x_eval.shape[0]) 
    print score
    if score[1] >= bscore:    
	#model.save(MODEL)
	model.save_weights(MODEL+'.we')
	bscore=score[1]
    return bscore
train_eval()
