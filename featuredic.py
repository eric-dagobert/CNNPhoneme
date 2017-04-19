import collections
import sys
import io

import cPickle
from collections import defaultdict
import ipapy
from ipapy import arpabetmapper
from ipapy.arpabetmapper import ARPABETMapper

_amapper = ARPABETMapper()

_feat1 = {'vowel': 1,'consonant':10}
_featvowel1 = {'front':1,'near-front':2,'central':3,'near-back':4,'back':5}
_featvowel2 = {'close':1,'near-close':2,'close-mid':3,'mid':4,'open-mid':5,'near-open':6,'open':7}
_featvowel3 = {'unrounded':1,'rounded':2}
_featvowel4 = {'rhotacized':1}

_featcons1 = {'bilabial':1,'labio-dental':2,'linguo-labial':3,'dental':4,'alveolar':5,'palato-alveolar':6,'retroflex':7,'alveolo-palatial':8,'palatal':9,'velar':10,'uvular':11,'epiglottal':12,'glottal':13,'labio-velar':14}
_featcons2 = {'nasal':1,'plosive':2,'sibilant-affricate':3,'non-sibilant-affricate':4,'sibilant-fricative':5,'non-sibilant-fricative':6,'approximant':7,'flap':8,'trill':9,'lateral-affricate':10,'laterale-fricative':11,'lateral-approximant':12,'lateral-flap':13}
_featcons3 = {'voiced':1,'voiceless':2}

_phonetov = defaultdict()
#f = None
f = open('phonetovec4.npy', mode='rb')
if f is not None:
    _phonetov = cPickle.load(f)
    f.close()
                
def phonetovec(u):
    l = []
    if u == 'sil' or u is None: return [0,0,0,0]
    if _phonetov.has_key(u):
        return _phonetov[u]
    for kx, x in _amapper.items():
        if x == u.upper():    
            print kx[0]
            if   'consonant' in kx[0]:
                l = [10]
                flist = [_featcons1,_featcons2,_featcons3]
                tlen=4
            else:
                l  =[1]
                flist = [_featvowel1,_featvowel2,_featvowel3,_featvowel4]
                tlen=5
            for i,dic in enumerate(flist):
                l.append(0)
                for k in dic.keys():
                    for t in kx[0].split():
                        if t == k:
                            l[i+1] = dic[k]
                            if l[0]==10 : l[i+1]+=10
                            break
            
            if len(l)==4:
                l.append(20)
            l = l[1:]            
            if 0 in l[:-1]:
                print 'error'
            _phonetov[u]=l
            f = open('phonetovec.npy', mode='wb')
            cPickle.dump(_phonetov, f)
            f.close()
            return l
def phonetovec5(u):
    l = []
    if u == 'sil': return [0,0,0,0,0]
    if _phonetov.has_key(u):
        return _phonetov[u]
    for kx, x in _amapper.items():
        if x == u.upper():    
            print kx[0]
            if   'consonant' in kx[0]:
                l = [10]
                flist = [_featcons1,_featcons2,_featcons3]
                tlen=4
            else:
                l  =[1]
                flist = [_featvowel1,_featvowel2,_featvowel3,_featvowel4]
                tlen=5
            for i,dic in enumerate(flist):
                l.append(0)
                for k in dic.keys():
                    for t in kx[0].split():
                        if t == k:
                            l[i+1] = dic[k]
                            if l[0]==10 : l[i+1]+=10
                            break
            
            if len(l)==4:
                l.append(20)
                        
            if 0 in l[:-1]:
                print 'error'
            _phonetov[u]=l
            f = open('phonetovec.npy', mode='wb')
            cPickle.dump(_phonetov, f)
            f.close()
            return l

def fold_phones(ph):
    if ph in [u'iy']: return 'ix'
    if ph in [u'hh',u'hv']: return 'hh'
    if ph in [u'epi'] : return None
    if ph in [u'cl',u'pcl',u'tcl',u'kcl',u'qcl'] : return None
    if ph in [u'vcl',u'bcl',u'dcl',u'gcl'] : return None
    if ph in [u'epi'] : return None
    if ph in [u'sil',u'h#',u'#h',u'pau'] : return 'sil' # return u'pau'
    return ph
 
def fold_phones2(ph):
    
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

#print phonetovec('r')
#print phonetovec('er')
#print phonetovec('nx')
#print phonetovec(  'dx')
