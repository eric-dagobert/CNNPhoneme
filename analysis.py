from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import librosa
from librosa import display,feature,util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm as CM
from matplotlib  import mlab
from pylab import *
import sklearn
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
from keras import utils
import featuredic
import operator
_phoneconf = defaultdict()

def loadconf():
    global _phoneconf
    f = open('phoneconf.npy','rb')
    _phoneconf = cPickle.load(f)
    f.close()
    
def addconf(predict, target):
    if _phoneconf.has_key((predict,target)):
        _phoneconf[(predict,target)] += 1
    else:
        _phoneconf[(predict,target)]=1
    f = open('phoneconf.npy','wb')
    cPickle.dump(_phoneconf,f)
    f.close()

def cosph(v1,v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1 if n1 == n2 else  0
    return np.dot(v1,v2)/n1/n2
 
def closeness (v,t):
    return np.where(v==t)[0].shape[0]
def closest(v,y):
    msim = 0
    tt = []
    kk = None
    normdiff = inf
    print 'source:',v
    for k,t in y:
        sim = closeness(v,t) #np.where(v==t)[0].shape[0]
        diff = np.linalg.norm(v-t)
        if sim > msim:
            msim = sim
            normdiff = diff
            tt = t
            kk =k
            print 'candidate',k,t,sim,'distance',diff 
        else:
            if sim == msim and diff <= normdiff :
                msim = sim
                normdiff = diff
                tt = t
                kk =k
                print 'candidate',k,t,sim,'distance',normdiff
    return tt,kk,msim
        
def vectophone(v,p2):
    if (np.where(v == 0))[0].shape[0] >3 :
        return [0,0,0,0],'sil'
    for k,l in featuredic._phonetov.items():
        if (l == v).all():
            return l,k
    print ' target :',p2
    tt,k,sim = closest(v, featuredic._phonetov.items())
    print 'closest',k
    return tt,k

def phoneprobs(success=False):
    l = []
    for (z,v) in sorted(_phoneconf.items(),key=operator.itemgetter(1),reverse=True):
        s = sum([x[1] for x in predicted(z[1])])
        if s == 0:
            continue
       
        if success and z[0]==z[1]:       
            l.append((z[0],v/s))
        elif not success and z[0] != z[1]:
            l.append(((z[0] ,z[1]),v/s))
    for z in sorted(l,key=operator.itemgetter(1),reverse=True):
        print z,featuredic.phonetovec(z[0][0]), featuredic.phonetovec(z[0][1])
    return l
def predicted(target):
    l= []
    for k in _phoneconf.keys():
        if k[1] == target:
            l.append((k[0],_phoneconf[k]))
    return sorted(l,key = operator.itemgetter(1),reverse=True)


def analyze():
    y = np.load('y_e36.npy')
    x = np.load('x_e36.npy')
    cnn = keras.models.load_model('model_convect5d36.kn')
    yp = cnn.predict(x, batch_size=x.shape[0])
    count = 0
    nn=0
    c = [0,0,0,0,0]

    for y_,yt in zip(yp,y):
        yvec,ytvec = np.argmax(y_,axis=1),np.argmax(yt,axis=1)
        _,p2 = vectophone(ytvec,'')
       
        v, p1 = vectophone(yvec,p2)
        sim = np.where(v == ytvec)[0].shape[0]
        p1 = featuredic.fold_phones2(p1)
        p2 = featuredic.fold_phones2(p2)
        if p1 == p2:
            sim=4
            nn+=1
        distance = np.linalg.norm(v-ytvec)
        if distance == 0:
            sim = 4
        elif distance < 4:
            sim = 3
        c[sim] += 1
        count += 1
        addconf(p1,p2)
        if p1 is None:
            nn+=1
            print yvec, ytvec
        if p1 != p2 :
            print 'predict:',p1,'true:',p2
            print 'diff >>'
            print v
            print ytvec
            print 'sim' ,sim,'distance',np.linalg.norm(v-ytvec)
            print '-----'
    print c,nn, count
    print '<=40%      40%        60%         80%         100%:'
    print [x /count for x in c]

analyze()
phoneprobs(False)
#loadconf()


