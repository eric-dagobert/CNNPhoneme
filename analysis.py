#!/usr/bin/python -O
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
#import librosa
#from librosa import display,feature,util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm as CM
from matplotlib  import mlab
from pylab import *
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import discriminant_analysis
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn import cluster
import scipy
from scipy.io import wavfile
from collections import defaultdict
import nltk
import cPickle
from nltk import corpus
from nltk.corpus import timit
from sklearn import decomposition
global HEIGHT
global WIDTH
import ipapy
from ipapy import arpabetmapper
from ipapy.arpabetmapper import ARPABETMapper
import keras
#import cnn2
import os
from os import path
from keras import utils
import featuredic
import operator
import libnet
import io
_phoneconf = defaultdict()


#POSTFIX='extph.npy'
#MODEL='model_cv2.we'
POSTFIX='extphtd.npy'
MODEL='model_cv2td.we'
MAXSIZEDIM=[0,0,0,0,0]
_phonedict = defaultdict()

import libnet
CLOSE=0       
POTENTIAL = 0
def test_acc(x,ypt,cnn):
    yt = np.load('y_e' + POSTFIX)
    
    yy = np.argmax(ypt,axis=2)
    a = np.where(yt==yy)
    sc = cnn.evaluate(x,keras.utils.to_categorical(yt, num_classes=52).reshape(-1,5,52),batch_size=x.shape[0])
    print 'score',sc
    print 'acc=', a[0].shape[0]/(5*yt.shape[0])
    print 'mse=', np.mean(np.linalg.norm(yt-yy,axis=1))
    #c = input(prompt='>')

def testfiles():
    x = np.load('x_t'+POSTFIX)
    cnn= libnet.rnnv1(64,40, 52,True)
    cnn.load_weights(MODEL)
    cnn.compile(loss='categorical_crossentropy',
	          metrics=['categorical_accuracy'],
	          optimizer='adam')    
    
    #cnn = keras.models.load_model(MODEL)
    yp = cnn.predict(x, batch_size=1000)
    fname = 'ypredtrain.npy'
    f = open(fname,'wb')
    cPickle.dump(yp,f)
    f.close()
    xe = np.load('x_e'+POSTFIX)
    ypt = cnn.predict(xe, batch_size=1000)
    fname = 'ypredtest.npy'
    f = open(fname,'wb')
    cPickle.dump(ypt,f)
    f.close()
    test_acc(xe,ypt,cnn)
    

def  classifier(rf=False):
    
    if rf:
	testfiles()
    rf = DecisionTreeClassifier(max_depth=10)
    ytrain = np.load('y_t'+POSTFIX)
    fname = 'ypredtrain.npy'
    f = open(fname,'rb')
    daxis = 2 
    yout_cnn = cPickle.load(f)
    f.close()
    yout_cnn = yout_cnn.reshape((ytrain.shape[0],-1))
    fname = 'ypredtest.npy'
    f = open(fname,'rb')
    yeval_cnn = cPickle.load(f)
    f.close()
    rf.fit(yout_cnn,ytrain)
    yeval = np.load('y_e'+POSTFIX)
    return rf,yeval_cnn,yeval


def cosph(v1,v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1 if n1 == n2 else  1000
    return np.dot(v1,v2)/n1/n2
 
def closeness (v,t):
    #dif = sum([abs(v[i]-t[i])*10**i for i in range(4)])
    vt = (v-t)
    mtch = np.where(v!=t)[0].shape[0]+1
    return np.abs(np.linalg.norm(v-t)*cosph(v,t))
   
def closest(v,verbose=False):
    global POTENTIAL
    msim = 1e10
    tt = []
    kk = None
    normdiff = inf
    pot = []
    for k,t in featuredic._phonetov.items():
	tv = np.array(t)
        sim = closeness(v,tv) #*featuredic.phonefreq(t)#np.where(v==t)[0].shape[0]
	p = featuredic.phonefreq(t)
	pot.append((t,k,sim,p))
	if verbose:
	    print 'candidate',k,t,'distance',sim ,'wprob',p    
    cand =sorted(pot, key=operator.itemgetter(2),reverse=False)
    best = cand[0]
    return np.array(best[0]),best[1]

def match_vowcons(yp):
    yprd = np.argmax(yp,axis=1)
    if  np.where(yprd <20)[0].shape[0] >= 3:
	if yprd[0] >= 20:
	    yprd[0] =1
	    for i in range(1,5):
		if yprd[i]  > 20 : 
		    yprd[i] -= 20
    elif np.where(yprd >=20)[0].shape[0] >= 3:
	if yprd[0] < 20:
	    yprd[0] = 20
	    for i in range(1,5):
		if yprd[i]  < 20 : 
		    yprd[i] += 20
    
def accuracy(rf = False,verbose=False):
   
    _correctphone = defaultdict()     
    bitmatch = 0
    fullmatch = 0
    acctree_per_cat = np.zeros((5,))
    accclass_per_cat = np.zeros((5,))
    
    #best=0
    rf,ycnn,ytrue = classifier(rf)
    ncountpred=0
    ncounttree=0
    nf = 0
    p = 'dum'
    pclass = 'dum'
    totph = 0
    for i,(yp,yt) in enumerate(zip(ycnn,ytrue)):
	yprd = match_vowcons(yp)
	yprd = np.argmax(yp,axis=1)
	p = featuredic.cattophone(yprd)	    
	pt =featuredic.cattophone (yt)
	if pt == 'not found':
	    print 'problem'
	
	#if featuredic._phonefreq[pt] <= 64:
	 #   continue
	totph +=1
	if not  _correctphone.has_key(pt):
	    _correctphone[pt]=(0,0)
	
	ytree = np.argmax(yp,axis=1)
	if p != pt:
	    ytree = rf.predict(yp.reshape((1,-1)))[0]
	    yprd ,p= closest(yprd)
	totalp = _correctphone[pt]
	totalp = (totalp[0],totalp[1]+1)
	#pclass = 'tree classifier'
	
	ptree = featuredic.cattophone(ytree)
	
	if ptree == 'not found': 
	    nf +=1
	    ytree,ptree = closest(ytree)
	  
	if ptree == pt:
	    totalp= (totalp[0]+1,totalp[1])
	_correctphone[pt]=totalp
	
	best=0
	ncount = 0
	
	nclass =  5
	pr = [0]*nclass
	if ptree == pt:
	    totalp =(totalp[0]+1,totalp[1])	
	mm = np.where(yt ==yprd)[0]
	mm2 =  np.where(yt ==ytree)[0]
	if mm.shape[0]==nclass:ncountpred+=1
	if mm2.shape[0]==nclass:ncounttree+=1
	
	acctree_per_cat[mm2] += 1
	accclass_per_cat[mm] += 1
	if  not _phonedict.has_key(pt):
	    _phonedict[pt] = [mm2.shape[0],nclass]
	else:
	    _phonedict[pt] = [_phonedict[pt][0] + mm2.shape[0],_phonedict[pt][1] + nclass]
	ncount = mm.shape[0]
	bitmatch += ncount
	maxcount = len(yt) 
	if ncount == maxcount:
	    fullmatch += 1
	    if verbose: 
		print 'correct',p,'(',pt,pclass,')'
	elif verbose:
	    print '================='
	    print 'target',pt,'predict',p,' - ', pclass
	    print '- -- - - - - - - - - - - - - - - - - - - -'
	    print 'true' ,yt
	    print 'predict ',yp 
	    print 'classifier',ytree
	    print '- -- - - - - - - - - - - - - - - - - - - -'
    print 'total ph',totph
    print 'fullmatch',fullmatch/totph,'bitmatch',bitmatch/totph/maxcount,'not found',nf/totph
    print '% decision tree',ncounttree/totph
    print '% predict',ncountpred/totph
    
    print 'per cat decision tree:',  acctree_per_cat/totph
    print 'per cat predict:',  accclass_per_cat/totph
    perc =[(x, _correctphone[x][0]/_correctphone[x][1],_correctphone[x][1]) for x in _correctphone.keys()]
    for x,p,v in sorted(perc,key=operator.itemgetter(1),reverse=True):
	print x,':',p,'(',v,')'


accuracy(False)


