#!/usr/bin/python -O
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
import featuredic
_ZERO=32
global _LOOKAT 
PLOT_=False
def pad(padvalue,s,twidth):
    pad1 = (twidth-s.shape[1])//2
    pad2 = twidth-pad1-s.shape[1]
    z1 = np.zeros((HEIGHT,pad1))
    z2 = np.zeros((HEIGHT,pad2))
    z1[:,:]=padvalue
    z2[:,:]=padvalue
    s1 = np.hstack((z1,s))
    s1 = np.hstack((s1,z2))    
    return s1
def plotone(freqs,bins,Z,title, subplot=None):
    z = 10*np.log10(Z)
    z = np.flipud(z)
    
    freqs += 0
    extent = 0,np.amax(bins),freqs[0],freqs[-1]
    if subplot is None :
        plt.figure()
        ax = plt.subplot(111)   
    else:
        ax = subplot#.gca()
    ax.imshow(z,extent=extent)
    ax.axis('auto')

def plotim(z,title="", subp=None,flip=False):
    # Z = 10*np.log10(z)

    Z=z
    if flip:
        Z= np.flipud(z)
    extent = 0,Z.shape[1],0,Z.shape[0]
    if subp is None:
        plt.figure()
        ax = plt.subplot(111)
    else:
        ax=subp
    #Z= z #np.flipud(z)
    im  = ax.imshow(Z,extent=extent)
    ax.axis('auto')
    ax.set_title(title)
    return im

def mfcc(signal,rate,nphones,pos,title):
    nfeatures = nphones*3
    nfft = 512
    z = python_speech_features.mfcc(signal=signal,samplerate=rate,nfft=nfft,nfilt=nfeatures, numcep=nfeatures//2,appendEnergy=True)
    d = python_speech_features.delta(z, 2)
    for x in pos:
        plotim(z[:,3*x:3*(x+1)],"mfcc "+title,False)
        plotim(d[:,3*x:3*(x+1)],"delta "+ title,False)
    plotim(z,'mfcc full',False)
    #plt.show()
    

def extract_features(sig,begin,endd, rate,nphones, size_one):
    nfft = 2*HEIGHT-1
    l= size_one*1.5*nphones
    L = endd-begin
    sig = sig[begin:endd]
    #L = len(sig)
    noverlap = nfft-int(round(L/l,0))
    if noverlap >= nfft:
        noverlap = nfft -1
    sp = mlab.specgram(sig, NFFT=nfft,Fs=rate,sides='default',scale_by_freq=True,noverlap=noverlap)
    spdata = sp[0]
    #plt.show()
    return sp

               


def plotspec(signal,rate,Y,zeroes,lwb,size_one):
    N=len(lwb)
    R = 2
    L =N//R
    #L=10
    f,ax = plt.subplots(nrows=R,ncols=L)
    for r in range(0,R):
        i,j = 0,0
        b = 0
        bt = 0
        for idx,e in enumerate(zeroes[:L]):
            et = lwb[idx]
            subplot = ax[r,j] 
            nphones = Y[bt:et].shape[0]
            title = "%s"%(nphones)

            subplot.set_title(title)

            plt.subplot(subplot,sharex=ax[r,j-1])
            data, freqs, bins,_ = extract_features(signal,b,e[0], rate, nphones,size_one)
            if r == 0:
                plotone(freqs,bins,data,subplot)
            else:
                proj  = np.linspace(0, data.shape[1], num=100*nphones, endpoint=False, dtype=int)
                h = bins[proj]
                pdata = data[:,proj]
                plotone(freqs,h,pdata,subplot)
            #print title, wpca.shape[0],wpca.shape[1]/nphones
            j+=1
            if j >= Y.shape[0]/R:
                j=0
                i+=1
            b = e[0]+e[1]
            bt = et+1
    f.set_size_inches(20,8)
    f.subplots_adjust(hspace=0.2)
    plt.title='aa'
    plt.show()         

    
    
def plotsp(x,nclusters,spdata,Y):
    x = np.sort(x)
    #plt.figure()
    f,ax = plt.subplots(nrows=1,ncols=nclusters)
    currl = -1 
    seqno=-1
    st = []
    for  l in x:
        if l  not in st:
            seqno+=1
            st.append(l)
            #xprime = longest(x, l)
            xs = spdata[:,x==l]
            title = Y[seqno]
            #xs = spdata[:,xprime[0]:xprime[0]+xprime[1]]
            plotim(xs,'%s'%title,subp=ax[seqno])

def plotsphones(spdata,Y):
    if len(Y) == 1:
        plotim(spdata[0,],'%s'%Y[0])
    else:    
        f,ax = plt.subplots(nrows=1,ncols=len(Y))
        for  l in range(0,spdata.shape[0]):
            title = Y[l]    
            im = plotim(spdata[l,],'%s'%title,subp=ax[l])
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)
    
    #plt.show()

def mfcc_featurespca(signal,start,end,rate):
    global WIDTH,HEIGHT
    
    pc = decomposition.PCA(n_components=WIDTH, whiten=False)
   
    frame_step = 0.005
    l = 0
    nfft = 2*HEIGHT -1 
    data = python_speech_features.mfcc(signal=signal[start:end],samplerate=rate,nfft=nfft,nfilt=WIDTH*2, numcep=WIDTH,appendEnergy=True,winlen=0.01,winstep=frame_step) 
    
    if data.shape[0] < HEIGHT:
        data = np.vstack((data,np.zeros((HEIGHT-data.shape[0],WIDTH))))
    else:
        data= pc.fit_transform(data.T).T
    
    zz = python_speech_features.delta(data,2)        
    data  = np.hstack((data,zz))
    if data.shape[0] < WIDTH or data.shape[1] < WIDTH:
        return None
    
    subd = data.reshape(1,-1,WIDTH) 
    return subd
def spec_features(signal,start,end,rate,splitflag=False):
    global WIDTH
    
    pc = decomposition.PCA(n_components=WIDTH, whiten=True)
                            
    data, freqs, bins  = extract_features(signal, start, end, rate, 1, WIDTH)

    data = np.log10(data)
    data[data==-np.inf]=-11
    
    if data.shape[1] < WIDTH:
        data = pad(-11, data, WIDTH)
    
    d1= pc.fit_transform(data)
    data = pc.inverse_transform(d1)
    proj = np.linspace(0, data.shape[1], num=WIDTH,endpoint=False,dtype=int)
    subd = data[:,proj].reshape(HEIGHT,WIDTH)   
    if splitflag:
        subd = np.concatenate((subd[:,0:WIDTH//3].reshape(1,HEIGHT,-1),
                               subd[:,WIDTH//3:2*WIDTH//3].reshape(1,HEIGHT,-1),
                               subd[:,2*WIDTH//3:WIDTH].reshape(1,HEIGHT,-1)),                                                                                                  
                              axis=0)
    return subd
    
 
     

def builddata(n=-1,plot=False):
    fulldata = None         
    dsize=0
    ydata = None
    global WIDTH,HEIGHT
    fulldata = None
    ydata = None
    uttids = nltk.corpus.timit.utteranceids()
    wavs = nltk.corpus.timit.fileids(filetype='wav')
    phlist = []
    N = 0
    sumw = 0
    for i,utt in enumerate(uttids):
        print "utt %d of %d"%(i,len(uttids))
        if n > -1 and i >= n:
            break
        phones = nltk.corpus.timit.phone_times(utt)
        path = nltk.corpus.timit.abspath(wavs[i])
        (rate,signal)= scipy.io.wavfile.read(str(path))
        sentdata = None
        for prev,ph,ne in  zip(phones[:-2],phones[1:-1],phones[2:]):
            ph1,ph2,ph3 =featuredic.fold_phones2(prev[0]),featuredic.fold_phones2(ph[0]),featuredic.fold_phones2(ne[0])
            offsetback = min(prev[2]-prev[1],ph[2]-ph[1])*.3
            offsetforw = min(ne[2]-ne[1],ph[2]-ph[1])*.2
            if ph1 is None:
                offsetback = min(prev[2]-prev[1],ph[2]-ph[1]) *.8 
            if ph3 is None:
                offsetforw = min(ne[2]-ne[1],ph[2]-ph[1])*.6
            start =int( ph[1] - offsetback)
            end = int( ph[2] + offsetforw)               
            vph1,vph2,vph3 = featuredic.phonetovec(ph1), featuredic.phonetovec(ph2), featuredic.phonetovec(ph3)      
            #if phone is None : continue
            phlist.append(end-start)            
            subd=None
            subd =spec_features(signal, start, end, rate)
            if subd is None :#or subd1 is None:
                continue    
            if vph2 is None:
                continue
            ydata =vph2 if ydata is None else np.vstack((ydata,[vph2]))
        
            fulldata = [subd] if fulldata is None else np.concatenate((fulldata,[subd]),axis=0)
            
        if plot :    
            plotsphones(sentdata, phones[1:-1])
            plt.show()
    return fulldata,ydata


HEIGHT=64
WIDTH=36
#PLOT_ = True

data,labels = builddata()

labels = labels.astype(float)

print 'data:',data.shape
print 'labels:',labels.shape, '[',labels.min(),labels.max(),']'
np.save('spec_6436mod', data)
np.save('spec_6436mod_vec',labels)
print 'saved in 6436mod'
    
#print _shapes

