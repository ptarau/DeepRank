from nltk.parse.corenlp import CoreNLPDependencyParser
import stanfordnlp
import os
import sys
from abc import ABC, abstractmethod
from timeit import timeit as tm

# nlp toolkit plugin - abstract class
class NLP_API(ABC):
  def __init__(self, text):
    self.text=text
    self.triples=None
    self.lemmas=None
    self.words=None

  @abstractmethod
  def get_triples(self):
    pass

  @abstractmethod
  def get_lemmas(self):
    pass

  @abstractmethod
  def get_words(self):
    pass

  def get_all(self):
    return self.get_triples(),self.get_lemmas(),self.get_words()

# subclass using Stanford coreNLP
class CoreNLP_API(NLP_API):
  def __init__(self, text):
    super().__init__(text)

    self.dparser = CoreNLPDependencyParser(url='http://localhost:9000')

    # gss is a list of graph generators with
    # number of elements equal to the number of sentences
    self.gss = list(self.dparser.parse_text(self.text))
 
    self.get_triples()
    self.get_lemmas()
    self.get_words()

  def get_triples(self):
    if not self.triples :
      self.triples = []
      for gs in self.gss:
        self.triples.append(list(gs.triples()))
    return self.triples

  def _extract_key(gss,key):
    wss = []
    for gs in gss:
      ns=list(gs.nodes.items())
      ws=[None]*(len(ns)-1)
      for k,v in ns :
        ws[k-1]=v[key]
      wss.append(ws)
    return wss

  def get_lemmas(self):
    if not self.lemmas:
      self.lemmas = CoreNLP_API._extract_key(self.gss,'lemma')
    return self.lemmas

  def get_words(self):
    if not self.words:
       self.words = CoreNLP_API._extract_key(self.gss, 'word')
    return self.words

# subclass using  torch-based  stanfordnlp - Apache licensed
class StanTorch_API(NLP_API):
  def __init__(self, text):
    super().__init__(text)
    self.doc=self.start_parser()

  def get_triples(self):
    if not self.triples:
      tss=[]
      for s in self.doc.sentences:
        ts=[]
        for dep_edge in s.dependencies:
          source=(dep_edge[0].text,dep_edge[0].pos)
          target=(dep_edge[2].text,dep_edge[2].pos)
          t= (source,  dep_edge[1], target)
          #print('DEPEDGE', len(dep_edge),t)
          ts.append(t)
        tss.append(ts)
        self.tuples=tss
    return self.tuples

  def get_words_and_lemmas(self):
    if not self.lemmas or not self.words :
      wss=[]
      lss=[]
      for s in self.doc.sentences:
        ws = []
        ls = []
        for w in s.words:
          ws.append(w.text)
          ls.append(w.lemma)
        wss.append(ws)
        lss.append(ls)
      self.words=wss
      self.lemmas=lss

  def get_words(self):
   self.get_words_and_lemmas()
   return self.words

  def get_lemmas(self):
   self.get_words_and_lemmas()
   return self.lemmas

  def start_parser(self):
    mfile = os.getenv("HOME") + \
            '/stanfordnlp_resources/en_ewt_models'
    sout=sys.stdout
    serr=sys.stderr
    f = open(os.devnull, 'w')
    sys.stdout = f
    sys.stderr = f
    # turn output off - too noisy
    if not os.path.exists(mfile):
      stanfordnlp.download('en',confirm_if_exists=True)
    nlp = stanfordnlp.Pipeline()
    dparser = nlp(self.text)
    # turn output on again
    sys.stdout=sout
    sys.stderr=serr
    return dparser

def apply_api(api,fname) :
  with open(fname,'r') as f:
    ls=f.readlines()
    text="".join(ls)
    return api(text).get_all()

def t1() :
  print('with coreNLP')
  print('')
  text='The happy cat sleeps. The dog just barks today.'
  p=CoreNLP_API(text)
  print(p.get_triples())
  print('')
  print(p.get_lemmas())
  print('')
  print(p.get_words())
  print('')
  print(p.get_triples())
  print('-'*50)
  print('')
  
def t2() :
  print('with stanfordnlp - torch based')
  print('')
  text = 'The happy cat sleeps. The dog just barks today.'
  p = StanTorch_API(text)
  print(p.get_triples())
  print('')
  print(p.get_lemmas())
  print('')
  print(p.get_words())
  print('')

# benchmark

def bm1(fname) :
  api=CoreNLP_API
  (ds,ls,ws)=apply_api(api,fname)
  print('coreNLP','sents=',len(ws))

def bm2(fname) :
  api=StanTorch_API
  (ds, ls, ws) = apply_api(api,fname)
  print('stanfordnlp','sents=',len(ws))

def bm() :
  fname = 'examples/const.txt'
  fname = 'examples/einstein.txt'
  fname = 'examples/tesla.txt'
  #print(tm(lambda: bm1(fname), number=1))
  print(tm(lambda: bm2(fname), number=1))

#t1()
#t2()
bm()
