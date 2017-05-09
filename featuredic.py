from __future__ import division
import collections
import sys
import io
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer,LabelEncoder
import cPickle
from collections import defaultdict
import ipapy
from ipapy import arpabetmapper
from ipapy.arpabetmapper import ARPABETMapper
import nltk
import numpy as np
from nltk import corpus
import operator
_amapper = ARPABETMapper()

_labels =['none','vowel', \
          'front','near-front','central','near-back','back',\
          'close','near-close','close-mid','mid','open-mid','near-open','open',\
          'unrounded','rounded',\
          'not-rhotacized','rhotacized','near-back_ext','near-close_ext',\

          'consonant',\
          'bilabial','labio-dental','linguo-labial','dental','alveolar','palato-alveolar',
          'retroflex','alveolo-palatial','palatal','velar','uvular','epiglottal','glottal','labio-velar',\

          'nasal','plosive','sibilant-affricate','non-sibilant-affricate','sibilant-fricative',\
          'non-sibilant-fricative','approximant','flap','trill','lateral-affricate','laterale-fricative','lateral-approximant','lateral-flap',\

          'voiced','voiceless',\
          'default-diac','diacritic-syllabic']


_invowcons = ['none','vowel','consonant']
_invow1 = ['none',  'front','near-front','central','near-back','back']
_invow2 = [ 'none','close','near-close','close-mid','mid','open-mid','near-open','open']
_invow3 = [ 'none','unrounded','rounded']
_invow4 = [ 'none','not-rhotacized','rhotacized','near-back_ext','near-close_ext']

_incons1 = ['none', 'bilabial','labio-dental','linguo-labial','dental','alveolar','palato-alveolar','retroflex','alveolo-palatial','palatal','velar','uvular','epiglottal','glottal','labio-velar']
_incons2 = ['none',   'nasal','plosive','sibilant-affricate','non-sibilant-affricate','sibilant-fricative',\
            'non-sibilant-fricative','approximant','flap','trill','lateral-affricate','laterale-fricative','lateral-approximant','lateral-flap']
_incons3 = ['none', 'voiced','voiceless']
_incons4 = [ 'none', 'default-diac','diacritic-syllabic']

_labcons = [_incons1,_incons2,_incons3,_incons4]
_labvow = [_invow1,_invow2,_invow3,_invow4]

_VO_OFFSETS=[(1,1),(2,12),(13,14),(15,16),(17,20)]
_CO_OFFSETS=[(21,21),(22,35),(36,48),(49,50),(51,52)]

_phonetov = defaultdict()
_phonefreq = defaultdict()
_reversephtv = defaultdict()
_phonetocat = defaultdict()

_MB = MultiLabelBinarizer(classes=_labels)
_LE = LabelEncoder()
_LE.fit(_labels)

def relabel(y):
   y_dim=y.copy()
   dimensions = [(0,0)]*5
   dimensions[0]=3
   for i,v in enumerate(y_dim):
      if y[i,0] == 1:
         labels = _labvow
         
      elif y[i,0]==20:
         v[0] = 2
         labels = _labcons
      else:
         v = [0,0,0,0,0]
         continue
      for dim in range(1,5):
         ind = labels[dim-1].index(_labels[y[i,dim]])
         v[dim]=ind
         dimensions[dim]=len(_labvow[dim-1]),len(_labcons[dim-1])
   return dimensions,y_dim

def unlabel(y):
   yout = y.copy()
   for i , v in enumerate(yout):
      if y[i,0]==2: 
         v[0] = 20
         l = _labcons
      elif y[i,0]==1:
         l = _labvow
      else:
         v = [0,0,0,0,0]
         continue
      for dim in range(1,5):
         ind = y[i,dim]
         v[dim]= _labels.index(l[dim-1][ind])
   return yout

def argmax_cat(vbin,index):
   vs,ve = _VO_OFFSETS[index]
   cs,ce = _CO_OFFSETS[index]
   v = np.argmax(vbin[vs:ve+1])+vs
   c = np.argmax(vbin[cs:ce+1])+cs
   if vbin[v] > vbin[c]:
      return v,c,True
   else:
      return v,c,False


def bin_to_vec(bins):
   l = []
   for c, b in enumerate(bins):
      if bins[c] == 1:
         l.append(c)
   return tuple(l)


def cleardicts():
   _phonetov.clear()
   _phonefreq.clear()
   _reversephtv.clear()

_lvow = set()
_lcons  = set()
_CALCFREQ = False
f = open('phonetovec.npy', mode='rb')
if f is not None:
   _phonetov = cPickle.load(f)
   f.close()
f = open('reverseptv.npy', mode='rb')
if f is not None:
   _reversephtv = cPickle.load(f)
   f.close()
f = open('phonetocat.npy', mode='rb')
if f is not None:
   _phonetocat = cPickle.load(f)
   f.close()
f = None
#f = open('phonefreq.npy', mode='rb')
if f is not None:
   _phonefreq = cPickle.load(f)
   f.close()
f = open('phonefreq.npy', mode='rb')
if f is not None:
      _phonefreq = cPickle.load(f)
f.close()

def cosvec(v):
   n1 = np.linalg.norm(v)
   if n1 == 0:
      return 100
   cos = np.round(np.dot(v,_REFVEC)/(n1*np.linalg.norm(_REFVEC)),5)
   return cos
def adjustipa(kx):
   ipa = kx[0].split()
   if 'consonant'in ipa :
      if len(kx) > 1:
         ipa.append('diacritic-syllabic')
      else:
         ipa.append('default-diac')
   
   elif 'vowel' in ipa:
      if not 'rhotacized' in ipa:
         ipa.append('not-rhotacized')		
      if len(kx) >1:
         ipa.remove('not-rhotacized')
         for ch in kx[1].split():
            chkey = ch + '_ext'
            if chkey in _labels:
               ipa.append(chkey)
               break		    
   else:
      print 'problem',ipa
   return ipa
def phonetovec(u):
   l = []
   global _l1,_CALCFREQ
   if _phonetov.has_key(u):    
      return _phonetov[u]
   if u == 'sil' or u is None: return [0]*5 
   flagcons = False
   for kx, x in _amapper.items():
      if x != u.upper():  
         continue
      ipa = adjustipa(kx)
      catlist = sorted([_labels.index(z) for z in ipa])
      keylab =tuple( catlist)
      if 0 in catlist:
         print 'problem'	
      if _reversephtv.has_key(keylab):
         if u != _reversephtv[keylab]:
            print 'collision'
      else:
         _reversephtv[keylab]=u
      _phonetov[u]=keylab
      f = open('phonetovec.npy', mode='wb')
      cPickle.dump(_phonetov, f)
      f.close()
      f = open('reverseptv.npy', mode='wb')
      cPickle.dump(_reversephtv, f)
      f.close() 
      return catlist
   
def phonefreq(vec):
   if not _phonefreq.has_key(vec):
      return 0
   return _phonefreq[vec]/sum(_phonefreq.values())

def  cattophone(vec):
   vec = np.array(vec)
   if vec.sum()==0:
      return 'sil'
   vec.sort()
   if _reversephtv.has_key(tuple(vec)):
      return _reversephtv[tuple(vec)]
   return 'not found'

def bintophone(vbin):
   if _reversephtv.has_key(tuple(vbin)):
      return _reversephtv[tuple(vbin)]
   return 'not found'    

def fold_phones_small(ph):
   if ph in [u'iy']: return 'ix'
   if ph in [u'hh',u'hv']: return 'hh'
   if ph in [u'epi'] : return None
   if ph in [u'cl',u'pcl',u'tcl',u'kcl',u'qcl'] : return None
   if ph in [u'vcl',u'bcl',u'dcl',u'gcl'] : return None
   if ph in [u'epi'] : return None
   if ph in [u'sil',u'h#',u'#h',u'pau'] : return 'sil' # return u'pau'
   return ph

def fold_phones(ph):

   if ph in [u'aa',u'ao']: return 'aa'
   if ph in [u'iy']: return 'ix'
   if ph in [u'l',u'el']: return 'l'
   if ph in [u'ah',u'ax',u'ax-h']: return 'ah'
   if ph in [u'hh',u'hv']: return 'hh'
   if ph in [u'uw',u'ux']: return 'ux'
   if ph in [u'ng',u'eng']: return 'eng'
   if ph in [u'er',u'axr']: return 'er'
   if ph in [u'm',u'em'] : return 'm'
   if ph in [u'n',u'nx',u'en','dx'] : return 'n'
   if ph in [u'sh',u'zh'] : return 'sh'
   if ph in [u'epi'] : return 'sil'
   if ph in [u'cl',u'pcl',u'tcl',u'kcl',u'qcl'] : return 'sil'
   if ph in [u'vcl',u'bcl',u'dcl',u'gcl'] : return 'sil'
   if ph in [u'epi'] : return 'sil'
   if ph in [u'sil',u'h#',u'#h',u'pau'] : return 'sil' # return u'pau'
   return ph

_folded = [u'aa',u'iy',u'el',u'ax',u'ax-h',u'hh',u'uw',u'ng',u'er',u'axr',u'em',u'nx',u'en',u'zh'] 

def gen(n=-1):
   global _CALCFREQ
   cleardicts()
   _CALCFREQ = True
   uttids = nltk.corpus.timit.utteranceids()
   for i,utt in enumerate(uttids):
      print "utt %d of %d"%(i,len(uttids))
      print nltk.corpus.timit.sents(utt)
      if n > -1 and i >= n:
         break
      phones = nltk.corpus.timit.phone_times(utt)
      for ph in phones:
         phf = fold_phones(ph[0])
         v = phonetovec(phf)
         if _CALCFREQ:
            if _phonefreq.has_key(phf):
               _phonefreq[phf] += 1
            else:
               _phonefreq[phf]  = 1
            f = open('phonefreq.npy', mode='wb')
            cPickle.dump(_phonefreq, f)
            f.close()                        


def testph(ph):
   for kx, x in _amapper.items():
      if x == ph.upper():
         print ph, kx
         break

   vec= _phonetov[ph]
   #vec = bin_to_vec(vectb)
   print ph,':'    
   print '>>>>>>>>>'
   print vec
   for x in vec:
      print _labels[x]
   print '<<<<<<<<'
def featwhere(y,ph):
   v = phonetovec(ph)
   n =np.all(y==v,axis=1)
   return n
   
   
def testlabel(ph1,ph2,ph3,ph4):
   vec=np.vstack((  phonetovec(ph1), phonetovec(ph2), phonetovec(ph3), phonetovec(ph4)))

   y_dim = vec.copy()
   for i in range(0,5):
      y_dim = relabel(i,y_dim,vec)
   print 'initial'
   print vec
   print 'relabeled'
   print y_dim


   y_dimu = unlabel(y_dim)
   print 'unlabeled'
   print y_dimu

TEST=0
_CALCFREQ=True
if TEST:
      #
      gen()
      f = sorted(_phonefreq.items(),key=operator.itemgetter(1),reverse=True)
      tot = sum(_phonefreq.values())
      print tot
      s = 0
      for i in f:   
         print i[0],':',i[1],i[1]/tot*100
         s += i[1]
      print 's=',s
      #    for f in ['y','g','ah','aa','eh','uh','ow','oh','m','n','eng','hh']:
      #        z = phonetovec(f)
      ##       print f,z,phonefreq(z)
      #print _labels
      
      #for f in ['y','g','ah','aa','eh','uh','ow','m','n','eng','hh']:  
      
      #for f in ['y','g','ah','aa']:     
      #testlabel('y','g','ah','aa')




