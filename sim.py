from nltk.corpus import wordnet as wn
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
def is_similar(u,Pu,v,Pv) :
  pu=pos2tag(Pu)
  pv=pos2tag(Pv)
  if pu and pv :
    mysim = sim1(u,pu,v,pv)
    wup = sim2(u,pu,v,pv) 
    avg=(wup+mysim)/2
    if avg > 0.7 : return True
    else: return False

def sim2(u,pu,v,pv) :
  m=0
  for i in wn.synsets(u,pu) :
    for j in wn.synsets(v,pv) :
      s=i.wup_similarity(j)
      if s : m=max(m,s)
  return m
      
def sim1(u,pu,v,pv) : 
  us=set(wn.synsets(u,pu))
  if not us : 
    u=wn.morphy(u,pu)
    if u:
      us=set(wn.synsets(u,pu))
  vs=set(wn.synsets(v,pv))
  if not vs : 
    v=wn.morphy(v,pv)
    if v:
      vs=set(wn.synsets(v,pv))
  hus=set()
  for x in us : hus=hus.union(set(x.hypernyms()))
  for x in us : hus=hus.union(set(x.hyponyms()))
  hvs=set()
  for x in vs : hvs=hvs.union(set(x.hypernyms())) 
  #for x in vs : hvs=hvs.union(set(x.hyponyms())) 
  us=us.union(hus)
  vs=vs.union(hvs)
  cs=us.intersection(vs)
  if cs :
    return sigmoid(len(cs))
  else :
    return 0
  
def pos2tag(pos) :
  if not pos :
    return None
  c=pos[0]
  if c is 'N' : return 'n'
  elif c is 'V' : return 'v'
  elif c is 'J' : return 'a'
  elif c is 'R' : return 'r'
  else : return None

# basic wordnet relations
  
def wn_hyper(k,w,t) :
  i=1
  for s in wn.synsets(w) :
    for h in s.hypernyms() :
      r = s2w(h,t)
      if r : 
        yield r
        i+=1
        if i >k : return

def wn_hypo(k,w,t) :
  i=1
  for s in wn.synsets(w) :
    for h in s.hyponyms() :
      r = s2w(h,t)
      if r : 
        yield r
        i+=1
        if i >k : return
 
def wn_mero(k,w,t) :
  i=1
  for s in wn.synsets(w) :
    for h in s.part_meronyms() :
      r = s2w(h,t)
      if r : 
        yield r
        i+=1
        if i >k : return

def wn_holo(k,w,t) :
  i=1
  for s in wn.synsets(w) :
    for h in s.part_holonyms() :
      r = s2w(h,t)
      if r : 
        yield r
        i+=1
        if i >k : return

#  less useful but potentially interesting for weak proximity
def wn_syn(k,w,t) :
  i=1
  for s in wn.synsets(w) : 
    r = s2w(s,t)
    if r and r != w : 
      yield r
      i+=1
      if i >=k : return
      
def wn_up_down(k,w,t) :
  i=1
  xs=set()
  for s in wn_hyper(10,w,t) :
    for h in wn_hypo(10,s,t) :
      if h and h != w and not h in xs: 
        xs.add(h)
        yield h
        i+=1
        if i >k : return

def wn_down_up(k,w,t) :
  i=1
  xs=set()
  for s in wn_hypo(10,w,t) :
    for h in wn_hyper(10,s,t) :
      if h and h != w and not h in xs : 
        xs.add(h)
        yield h
        i+=1
        if i >k : return
        
  
def s2w(s,t) :
  n = s.name()
  (w,tw,_) = n.split('.')
  if t==tw : return w.replace('_',' ')
  return None
  
      
      
######