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
_phoneconfdb = defaultdict()
_triphone= defaultdict()

def loadtri():
    global _triphone
    f = open('triphone.npy','rb')
    _triphone = cPickle.load(f)
    f.close()
def loadconf():
    global _phoneconf
    f = open('phoneconf.npy','rb')
    _phoneconf = cPickle.load(f)
    f.close()
    
def addtri(ph1,ph2,ph3):
    if _triphone.has_key((ph1,ph2,ph3)):
        _triphone[(ph1,ph2,ph3)] += 1
    else:
        _triphone[(ph1,ph2,ph3)] =1
    f = open('triphone.npy','wb')
    cPickle.dump(_triphone,f)
    f.close()

def addconf(predict, target):
    if _phoneconf.has_key((predict,target)):
        _phoneconf[(predict,target)] += 1
    else:
        _phoneconf[(predict,target)]=1
    f = open('phoneconf.npy','wb')
    cPickle.dump(_phoneconf,f)
    f.close()

def addconfdb(predict, target):
    if _phoneconfdb.has_key((predict,target)):
        _phoneconfdb[(predict,target)] += 1
    else:
        _phoneconfdb[(predict,target)]=1
    f = open('phoneconfdb.npy','wb')
    cPickle.dump(_phoneconfdb,f)
    f.close()

def cosph(v1,v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1 if n1 == n2 else  0
    return np.dot(v1,v2)/n1/n2
 
def closeness2 (v,t):
    if v[0]==1:       
        for i, (a,b) in enumerate(zip(v,t)):
            if a != b:
                if i == 0:
                    return 0
                if i in [1,2]: 
                    return -abs(a-b)
                if i == 3:
                    return 0 if a !=b else -100
    else:
        for i, (a,b) in enumerate(zip(v,t)):
            if a!=b : return -i
        return 4
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
        #diff = sum([(m)*10**(5-i) for i,m in enumerate(np.abs(v-t))])
        diff = np.linalg.norm(v-t)
        if sim > msim:
            msim = sim
            normdiff = diff
            tt = t
            kk =k
            print 'candidate',k,t,sim,'distance',diff 
        else:
            if sim == msim and diff <= normdiff :
            #if diff <= normdiff:
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
    if p2 == 'ah':
        print 'here'
    tt,k,sim = closest(v, featuredic._phonetov.items())
    print 'closest',k
    return tt,k

def test1():
    print vectophone(np.array([2,15,22,11,0]),'')
    print vectophone(np.array([0,10,2,0,0]))
    print vectofeat([0,2,2,0,0])
    print vectofeat([0,10,2,0,0])
    print vectofeat([0,4,2,0,0])
    

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
            if p1 == 'ow' and p2 == 'ah':
                print 'here'
            print 'diff >>'
            print v
            print ytvec
            print 'sim' ,sim,'distance',np.linalg.norm(v-ytvec)
            print '-----'
    print c,nn, count
    print [x /count for x in c]

def phone_prob(p2):
    l = 0
    s = sum(_triphone.values())
    for k in _triphone.keys():
        if k[1] == p2:
            l+=_triphone[k]/s
    return l


def predicted(target):
    l= []
    s = sum(_phoneconf.values())
    for k in _phoneconf.keys():
        if k[1] == target:
            l.append((k[0],log(_phoneconf[k]/s)))
    return sorted(l,key = operator.itemgetter(1),reverse=True)

def get_confdb():
    uttids = nltk.corpus.timit.utteranceids()#(sex='m')
    phlist = []
    N = 0
    sumw = 0
    ydata = None
    for i,utt in enumerate(uttids):
        print "utt %d of %d"%(i,len(uttids))
        phones = nltk.corpus.timit.phone_times(utt)
        for prev,ph,ne in  zip(phones[:-2],phones[1:-1],phones[2:]):
            ph1,ph2,ph3 =featuredic.fold_phones(prev[0]),featuredic.fold_phones(ph[0]),featuredic.fold_phones(ne[0])
            addtri(ph1,ph2,ph3)

#get_tri()

analyze()
loadconf()
loadtri()


