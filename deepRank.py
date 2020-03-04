from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser
import networkx as nx
from collections import defaultdict
import subprocess
from nltk.corpus import words
from nltk.corpus import stopwords
from graphviz import Digraph
from params import *
from sim import *

from params import *

def ppp(*args) : print(args)

def make_word_dict(fname) :
  wd=set(words.words())
  def add_from_file(fname) :
    added=True
    try :
      with open(fname,'r') as f:
        for w in f.readlines() :
          wd.add(w[:-1])
          #if w.startswith('space') : print(w[:-1],len(w))
    except :
      added=False
    return added
  if add_from_file('words.txt') :
    print('Added words.txt')
  if add_from_file(fname) :
    print('Domain-specific dictionary',fname,'added.')
  return wd


word_dict=make_word_dict('kb.txt')

stop_words=set(stopwords.words('english'))

# recognizers for for several word types



def isString(w) : 
  return isinstance(w,str)

def isWord(w) :
  if not isString(w) : return False
  #if isStopWord(w) : return False
  wl=len(w)
  return wl>1 and (isName(w) or inDict(w))
  
def isName(w) :
  return w.isalnum() and w[0].isupper()
  
def isStopWord(w) :
  return w.lower() in stop_words
  
def maybeWord(w) :
  if not isString(w) : return False
  clear_word = isWord(w) and len(w)>1
  return clear_word or isSpec(w) or isPunct(w) or w.isdigit() or hasDash(w)

def isSpec(w) :
  return w in ['a','A',"'s",'I'] or (w[0].isupper() and w.endswith('.'))

def isPunct(w) :
  return w in ",.?;:-'()" or w=='"'

def hasDash(w) :
  ws=w.split('-')
  if len(ws) < 2 : return False
  for w in ws:
    if ''==w or not maybeWord(w) : return False
  return True
  
def inDict(w) :
  def ends_well(w) :
    if w.endswith('ing') : return True
    if w.endswith('ed') : return True
    if w.endswith('ility') : return True
    if w.endswith('ly') : return True
    if w.endswith('er') : return True
    if w.endswith('st'): return True
    return False

  if w in word_dict : return True
  if w.capitalize() in word_dict: return True
  if w.lower() in word_dict : return True
  if ends_well(w) : return True
  if 's'==w[-1] :
    v=w[:-1]
    return ends_well(v)
  return False

# ensures the sentence does not have
# strings that do not make sense as keywords
# or words in a sentence
def isCleanSent(sent) :
  ok=True
  for w in sent :
    if maybeWord(w) : continue
    if inDict(w) : continue
    ok=False
    #ppp('UNCLEAN',w)
    break
  return ok
      
# recognize nouns, from their POS tags
def isNoun(tag) :
  #ppp('TAG=',tag)
  return tag[0]=='N'

# recognize adjectives

def isAdj(tag) :
  return tag[0]=='J'

# recognize verbs, from their POS tahs
def isVerb(tag) :
  return tag[0]=='V'  

# recognize subjects, from a relation label  
def isSubj(rel) :
  return rel.find('subj')>=0

# recognize obbjects, from a relation label 
def isObj(rel) :
  #return rel=='obj' or rel == 'dobj' or rel=='iobj'
  return rel.find('obj')>=0

def isMod(rel) :
  return rel=='nmod' or rel=='amod'

# sentece idetifiers are natural numbers
def isSent(s) :
  return isinstance(s,int)

# true for any argument
def isAny(x) : 
  return True

# class, building and ranking a text graph from a document or string  
class GraphMaker :
  def __init__(self) :
    # uses Stanford Parser on defaul port 9000 - see params.py
    self.dparser=CoreNLPDependencyParser(url=parserURL)
    self.clear()

  # clear saved state    
  def clear(self) :
    self.maxcc=None
    self.gs=None
    self.nxgraph=None
    self.ranked=None
    self.words2lemmas=set()
    self.noun_set=dict()
    self.svo_edges_in_graph=[]
  
  # digest a file  
  def load(self,fname) :
    self.clear()
    f=open(fname,'r')
    text=f.read()
    f.close()
    self.digest(text)

  def parse(self,text) :
      ts = self.dparser.parse_text(text)
      return list(ts)
  
  # digest a string using dependecy parser 
  def digest(self,text) :
    self.clear()
    chop=2**16
    gens=[]
    # deals with files that are too large to be parse at once
    while len(text) > chop:
      head=text[:chop]
      text=text[chop:]
      #ppp((head))
      if head :
        hs=list(self.parse(head))
        #ppp('PARSED')
        gens.append(hs)
    if gens :
      self.gs=[x for xs in gens for x in xs]
    else :
      self.gs=list(self.parse(text))
    #ppp('!!!',self.gs)
  
  # sentence as sequence of words generator   
  def sentence(self) :
    for g in self.gs :
      yield str.join(' ',list(gwords(g)))
     
  def wsentence(self) :
    for g in self.gs:
     ts= tuple(gwords(g))
    yield ts

  def nth_sent_words(self,n) :
    ws=tuple(gwords(self.gs[n]))
    return ws

  # sentence as sequence of lemmas generator  
  def lsentence(self) :
    for g in self.gs :
      yield tuple(glemmas(g))
   
# curates, reverses and adds some new edges 
 # yields an <edge, sentence in which it occurs> pair 
  def edgesInSent(self) :
    self.svo_edges_in_graph=[]

    def noun_to_def(x,tx,k) :
      if noun_defs :
        k_ = self.noun_set.get(x)
        if k==k_ :
          yield (x,tx,'first_in',k,'SENT')

    def edgeOf(k,g) :
      d=w2l(g)
      merge_dict(self.words2lemmas,d)
      make_noun_set(g,self.noun_set,k)
      svo_edges_in_sent=[]
      for ts in g.triples() :
        #ppp('TRIPLES',ts)
        fr,rel,to = list(ts)
        lfrom,ftag = d[fr[0]]
        lto,ttag=d[to[0]]
        # vn is True it is an s->v or o->v link
        so = isSubj(rel) or isObj(rel)
        vn = isVerb(ftag) and isNoun(ttag) and so
        if rel=='punct' and ttag=='.': 
          # sentence points to predicate verb
          yield (k,'SENT','predicate',lfrom,ftag)
        elif vn :
          # collects vs and vo links to merge them later into svo
          svo_edges_in_sent.append((lfrom,ftag, rel,lto,ttag))
          yield lfrom,ftag, rel,lto,ttag # verb to noun
          yield k,'SENT','about',lto,ttag # sent to noun
          # all words recommend sentence
          #yield lfrom,ftag,'recommends',k,'SENT' # verb to sent - in elif !
          for e in noun_to_def(lto, ttag,k,) : yield e  # noun to sent
          if noun_self : yield lto, ttag, 'self', lto, ttag
        elif isNoun(ttag) : # e.g. nmod relation
          #ppp('x-->n',k,lfrom,ftag,rel,lto,ttag)
          yield lfrom,ftag,rel,lto,ttag
          for e in noun_to_def(lto, ttag,k,) : yield e  # noun to sent
          if noun_self : yield lto, ttag, 'self', lto, ttag
          #yield lfrom, ftag, 'recommends', k, 'SENT' # dependent of noun to sent
        else : # yield link as is
          yield lto,ttag, rel,lfrom,ftag
          # all words recommend sentence
          if all_recs : yield lto,ttag,'recommends',k,'SENT'
         
        # merge compound terms, make their parts recommend them
        if isNoun(ftag) and isNoun(ttag) and rel=='compound' :
           comp= lto + ' ' + lfrom
           yield lfrom,ftag,'fused',comp,ftag
           yield lto,ttag,'fused',comp,ttag
           for e in noun_to_def(comp,ttag,k) : yield e
           if noun_self: yield comp,ttag,'self',comp,ttag
      # collect svo relations
      self.svo_edges_in_graph.append(to_svo(k,svo_edges_in_sent))
    k=0
    for g in self.gs :
      for e in edgeOf(k,g) :
        # collects words at the two ends of e
        #self.addWordsIn(e)
        yield e,k 
      k+=1
    
  # yields  the edge. possibly for each sentence where is found
  def multi_edges(self) :
    for e,k in self.edgesInSent() :
      yield e

  def edges(self) :
    for e in set(self.multi_edges()) :
      yield e

  # returns final networkx text graph
  def graph(self) :
    if(self.nxgraph) : return self.nxgraph
    dg=nx.DiGraph()
    
    for e in self.edges() :
      f,tf,r,t,tt=e
      dg.add_edge(f,t,rel=r) 
      
    self.nxgraph=dg  
    #ppp('DG:',dg,'END')

    #ppp('NOUN_SET',self.noun_set)
    return dg

  def size(self):
    return (self.nxgraph.number_of_nodes(),self.nxgraph.number_of_edges())

  # ranks (unless ranked and stored as such) the text graph
  def pagerank(self,pers=None) :
    if self.ranked : return self.ranked
    g = self.graph()
    pr=self.runPagerank(g,pers)
    self.ranked=pr
    if not giant_comp : return pr
    ccs=list(nx.strongly_connected_components(g))
    lc = len(ccs)
    #ppp('LEN_COMPS', lc)
    if lc < 4:
      self.maxcc = max(ccs, key=len)
    return pr

  def rerank(self,pers=None):
    self.ranked=None
    #ppp("RERANK",len(pers))
    return self.pagerank(pers=pers)

  # extracts best k nodes passing filtering test  
  def bestNodes(self,k,filter) :
    #g=self.graph()
    #comps=list(nx.strongly_connected_components(g))

    pr=self.pagerank()
    i=0
    ns=[]  # not a set - that looses order !!!
    for x,r in pr.items() :
      if i>=k : break
      #ppp('RANKED',x,r)
      if filter(x) :
        #ppp('FILTERED',x,r,'MC')
        if not self.maxcc or x in self.maxcc :
          if not x in ns :
            ns.append(x)
            i+=1
    return ns
   
  # specialization returning all best k nodes    
  def bestAny(self,k) :
    return self.bestNodes(k,lambda x: True)  
  
  # specialization returning best k sentence nodes    
  def bestSentencesByRank(self,k,filter=isAny) :
    best = self.bestNodes(100+k,isSent)
    if not best : return
    #ppp('BEST SENTS:',best)
    c=0
    for i in best :
      if not filter(i) : continue
      g=self.gs[i]
      lems = [w for w in glemmas0(g)]
      #ppp('LEMS',lems)
      if isCleanSent(lems) :
          sent = list(gwords(g))
          #sent=str.join(' ',list(gwords(g)))
          yield (i,sent)
          c+=1
      #else : ppp('SENT UNCLEAN',i)
      if c>=k : break

  def bestSentences0(self,k,filter=isAny):
     for i_ws in sorted( self.bestSentencesByRank(k,filter=filter) ) :
       yield i_ws

  def bestSentences(self,k,filter=isAny,with_word_graph=abstractive):
    xs=self.bestSentences0(k, filter=isAny)
    if with_word_graph=='yes' :
      xs = list(xs)
      pr = self.pagerank()  # make pr an (ordered) dict
      wg = nx.DiGraph()
      fs=set()
      ts=set()
      for s_ws in xs:
        _,ws=s_ws
        if not ws or len(ws) < 5: continue
        f=ws[0]
        t=ws[-1]
        if not f or not t : continue
        fs.add(f)
        ts.add(t)

        word_graph(wg,pr,s_ws)
        #print('LLLL',wg.number_of_edges())
      gshow(wg)
      for f in fs :
        if f not in wg.nodes(): continue
        for t in ts :
          if t not in  wg. nodes() : continue
          ps=nx.dijkstra_path(wg,f,t)
          if ps: print("DPATH",ps)

    for s_ws in xs:
      yield s_ws

  # specialization returning best k word nodes
  def bestWords(self,k) :
    #ppp('NOUNS',self.noun_set)
    c=0
    best=self.bestNodes(100+k,maybeWord)
    #ppp('BEST WORDS:',best)
    for w in best :
      if c>=k : break
      if not isStopWord(w) and self.hasNoun(w) : 
        yield(w)
        #ppp('BWORD',w)
      c+=1
  
  # true if a phrase has a noun in it    
  def hasNoun(self,w) :
    ws=w.split(' ')
    for v in ws:
      if v in self.noun_set : return True
    return False
    
  # runs PageRank on text graph  
  def runPagerank(self,g,pers) :
    #if pers : ppp('PERSONALIZED',pers)
    d = nx.pagerank(g,personalization=pers)
    #ppp("PR",d)

    # normalize sentence ranks by favoring those close to average rank
    sents = list(self.wsentence())
    lens = list(map(len,sents))
    #ppp('LENS:', lens)
    avg = sum(lens) / len(lens)

    #ppp('AVG SENT LENGTH:', avg)

    # reranks long sentences
    i = 0
    for ws in sents : # this makes i going over sentence ids
      #ppp('WS:',ws)
      if i in d :
        l=len(ws)
        r=d[i]
        newr = adjust_rank(r,l,avg)
        d[i]=newr
        #if l<6 : ppp(r,'--->',newr,l,'ws=',ws)
        i+=1

    sd = sorted(d,key=d.get,reverse=True)
     
    return dict((k,d[k]) for k in sd)
  
  # extracts k highest ranked SVO triplets  
  def bestSVOs(self,k) :
    rank_dict=self.pagerank()
    #ppp('PRANK',rank_list)
    ranked=[] # should not be a set !
    for rs in self.svo_edges_in_graph :    
      for r in rs :
        #ppp('SVO',r)
        (f,_),(rel,_),(t,_),sent_id=r
        srank=rank_dict[f]
        orank=rank_dict[t]
        if srank and orank :
          sorank = (2*srank+orank)/3 
          ranked.append((sorank,(f,rel,t,sent_id)))
    ranked=sorted(ranked,reverse=True)
    i=0
    exts=set()
    seen=set()
    for (_,e) in ranked :
      i+=1
      if i>k : break
      #ppp('SVO_EDGE',e)
      if e in seen : continue
      seen.add(e)
      yield e
      for xe in self.extend_with_wn_links(e,rank_dict) :
        f,_,t,_=xe
        if wn.morphy(f.lower()) != wn.morphy(t.lower()) :
          exts.add(xe)
    i=0
    for xe in exts :
       i+=1
       if i>k : break
       #ppp('XE',xe)
       yield xe
  
  # adds wordnet-derived links to a dictionary d
  # we tag them with is_a or part_of
  def extend_with_wn_links(self,e,d) :
    s,v,o,sent_id=e
    m=1 # how many of each are taken
    for x in wn_holo(m,s,'n') :
      if x in d : yield (s,'part_of',x,sent_id)
    for x in wn_mero(m,s,'n') :
       if x in d : yield (x,'part_of',s,sent_id)
    for x in wn_hyper(m,s,'n') :
      if x in d : yield (s,'is_a',x,sent_id)
    for x in wn_hypo(m,s,'n') : 
      if x in d : yield (x,'is_a',s,sent_id)
    for x in wn_holo(m,o,'n') :
      if x in d : yield (o,'part_of',x,sent_id)
    for x in wn_mero(m,o,'n') :
      if x in d : yield (x,'part_of',o,sent_id)
    for x in wn_hyper(m,o,'n') :
      if x in d : yield (o,'is_a',x,sent_id)
    for x in wn_hypo(m,o,'n') : 
      if x in d : yield (x,'is_a',o,sent_id)
   
  # visualize filtered set of edges with graphviz
  def toDot(self,k,filter,svo=False,show=True,fname='textgraph.gv') :
    dot=Digraph()
    g=self.graph()
    best=self.bestNodes(k,filter)
    for f,t in g.edges() :
      if f in best and t in best :
        dot.edge(str(f),str(t)) 
    if svo :
      svos=set()
      for (s,v,o,_) in self.bestSVOs(k) :
        svos.add((s,v,o))
      for e in svos :
        s,v,o=e
        dot.edge(s,o,label=v)      
    showGraph(dot,show=show,file_name=fname)      
   
  # visualize filtered set of edges as graphviz dot graph 
  def svoToDot(self,k) :
    dot=Digraph()
    for e in self.bestSVOs(3*k) :
      s,v,o=e
      dot.edge(s,o,label=v)
    showGraph(dot)  
       
      
  # specialize dot graph to words  
  def wordsToDot(self,k) :
    self.toDot(k,isWord)
  
  # specialize dot senteces graph words    
  def sentsToDot(self,k) :
    self.toDot(k,isSent)
    
  # visualize mixed sentence - word graph
  def allToDot(self,k) :
    self.toDot(k, lambda x: True)

# yields cleaned-up words of sentences in g    
def gwords(g) :
  for (i,w,l,p) in gsent(g) :
    yield w

def fix_par(w) :
    u=w.upper()
    if u=='-LRB-':
      return '('
    elif u=='-RRB-' :
      return ')'
    elif u=='-LSB-':
      return '['
    elif u=='-RSB-' :
      return ']'
    else :
      return w

# trims all bat lemmas and thir tags    
def glemmas(g) :
  for (i,w,l,p) in gsent(g) :
    yield (l,p)

# trims all bat lemmas 
def glemmas0(g) :
  for (i,w,l,p) in gsent(g) :
    yield l

# returns list of position,word,lemma,POS-tag tuples    
def gsent(g) :
    ws=[]
    for v in g.nodes.values() :     
      i = v.get('address')
      w=v.get('word')
      l=v.get('lemma')
      p=v.get('tag')
      if(w) : ws.append((i,fix_par(w),fix_par(l),p))
    ws.sort()
    return ws

def pers_dict(qgm) :
  return dict(
     (w,r) for (w,r) in qgm.pagerank().items()
     if maybeWord(w) and not isStopWord(w)
  )

def sent_words(gm):
    ctr = 0
    for g in gm.gs:
      yield (ctr,list(gwords(g)))
      ctr += 1

def word_orbit(gm) :
  orbit=defaultdict(list)
  for (s,ws) in sent_words(gm) :
    for w in ws:
      orbit[w].append(s)
  return orbit

def word_graph(g,pr,s_ws) :
    pr=dict(pr)
    s,ws0=s_ws
    ws=[w for w in ws0 if not isStopWord(w)]
    l=len(ws)
    for i in range(0,l-1) :
      f=ws[i]
      t=ws[i+1]
      r1 = pr.get(f)
      if not r1: r1=pr.get(f.lower())
      r2 = pr.get(t)
      if not r2: r2=pr.get(f.lower())
      if not r1: r1 = 0
      if not r2: r2 = 0
      r = 0
      if not r1 :  r=r2
      elif not r2 : r=r1
      else :
        r=(r1+r2)/2 # we will be looking for shortest
      if r==0 : r=float('inf')
      else : r = math.exp(1-r)
      g.add_edge(f,t,weight=r)
      #print("EDGE",(f,t,r))


# returns a dict of lemmas for word nodes in in g 
def w2l(g) :
  d=dict()
  for v in g.nodes.values() : 
    w=v.get('word')
    l=v.get('lemma')
    p=v.get('tag')
    if(w) : d[w]=(l,p)
  return d
  
# adds to given dict d nouns in nodes of g
def make_noun_set(g,d,k) :
  for v in g.nodes.values() :    
    w=v.get('lemma')
    p=v.get('tag')
    if(v) and isNoun(p) :
      if not w in d :
        d[w]=k
        #ppp('NOUN',w,k)
  
def merge_dict(tuples,d) :
   for k in d :
    l,lt=d[k]
    #ppp(l,lt)
    tuples.add((k,l,lt))

# turns into an svo triple two s and o links a verb points to    
def to_svo(k,rs) :
  #ppp('TRYING SVO',k,rs)
  s=dict()
  o=dict()
  svo=[]
  for r in rs:
    f,ft,rel,t,tt=r
    if not (isWord(f) and isWord(t)) : continue
    if isSubj(rel) :
      s[(f,ft)]=(t,tt)
    elif isObj(rel):
      o[(f,ft)]=(t,tt)  
    #else : ppp(k,'*** UNEXPCTED REL',f,t,rel)
  for vt in s :
    if vt in o :
      svo.append((s[vt],vt,o[vt],k)) # k=sent where found
  #for x in svo : ppp(k,'GOT SVO',x)
  return svo
  

# shows wk summaries and sk keywords from file
# extracts highest ranked dk svo relations  and visualizes 
# dk highest ranked filtered word to word edges as dot graph
# if svo optional arg is set to True, adns svo links to the graph
def runWithFilter(fileName,wk,sk,dk,vk,filter,show=pics=='yes') :
  gm= GraphMaker()
  gm.load(fileName)
 
  #for g in gm.gs : ppp(g)
  #ppp(list(gm.sentence()))
  #ppp(list(gm.lsentence()))
  #ppp(list(gm.edges()))
  #for g in gm.gs : ppp(list(g.triples()))
  #ppp(gm.graph().edges())
  #for p in gm.pagerank().items() : ppp(p)
  print("------PROCESSING:",fileName,"----------")
  print('noun_defs = ',noun_defs)
  print('all_recs = ',all_recs)
  print('nodes:', gm.graph().number_of_nodes())
  print('edges:', gm.graph().number_of_edges())
  print('')
  print_keys(gm.bestWords(wk))
  print('SUMMARY')
  print_summary(gm.bestSentences(sk))
  print_rels(gm.bestSVOs(vk))
  dotName=trimSuf(fileName)+".gv"
  gm.toDot(dk,filter,svo=True,fname=dotName,show=show)
  return gm


# same, with default values
def runWith(fileName) :
  runWithFilter(fileName,12,8,20,lambda x: True)

def print_summary(xs) :
  for s in xs :  
    n,ws=s
    print(n,':',end='')
    for w in ws :
      print(' ',w,end='')
    print('')
  print('')

def print_keys(ws) :
  print('KEYPHRASES')
  for w in ws :
    print(w,';')
  print('\n') 

def print_rels(rs) :
  print('RELATIONS')
  for r in rs :
    for w in r :
      print(w,' ',end='')
    print('')
  print('\n') 

def showAllEdges(g,file_name='textgraph.gv') :
  dot=Digraph()
  for e in g.edges():
    #ppp('EDGE:',e)
    f,ft,rel,t,tt=e
    rl=ft+'_'+str(rel)+'_'+tt
    dot.edge(str(f), str(t),label=rl)
  showGraph(dot,file_name=file_name)

def query_edges_to_dot(qgm):
    showAllEdges(qgm, file_name='query_graph.gv')

    # displays textgraph with graphviz
def showGraph(dot,show=True,file_name='textgraph.gv')  :

  dot.render(file_name, view=show) 

def gshow(g,**kwargs) :
  dot = Digraph()
  for e in g.edges():
    #print('EEEE',e)
    f,t= e
    w=g[f][t]['weight']
    dot.edge(str(f), str(t), label=str(w))
  showGraph(dot)



# extracts file name from path
def path2fname(path) :
  return path.split('/')[-1]

def trimSuf(path) :
  return ''.join(path.split('.')[:-1])

def justFname(path) :
  return trimSuf(path2fname(path))

def pdf2txt(fname) :
  subprocess.run(["pdftotext", fname])

# utilities

def take(k,seq) :
  c=0
  for x in seq :
    if c>=k : return
    yield x
    c+=1

# tests

def test0() : # might talke 1-2 minutes
  runWithFilter('examples/tesla.txt',wk,sk,30,50,maybeWord)
  
def test1() :
  runWithFilter('examples/bfr.txt',wk,sk,30,50,maybeWord)

def test2() :
  wk, sk = 3,3
  runWithFilter('examples/hindenburg.txt',wk,sk,20,50,maybeWord)

def test3() :
  runWithFilter('examples/const.txt',wk,sk,20,50,maybeWord)
  
def test4() :
  gm=runWithFilter('examples/summary.txt',12,3,30,50,maybeWord)
  showAllEdges(gm)

def test5() :
  runWithFilter('examples/heaven.txt',wk,sk,30,50,maybeWord)
  
def test6() :
  runWithFilter('examples/einstein.txt',wk,sk,30,50,maybeWord)
  
def test7() :
  runWithFilter('examples/kafka.txt',wk,sk,20,50,maybeWord)
  
def test8() :
  gm=runWithFilter('examples/test.txt',wk,sk,20,50,isAny,show=False)
  showAllEdges(gm)


def test9() :
  runWithFilter('examples/relativity.txt',wk,sk,20,50,maybeWord)

def test10() :
  runWithFilter('examples/cats.txt',wk,sk,20,50,maybeWord)

def test11() :
  runWithFilter('examples/wasteland.txt',wk,sk,20,50,maybeWord)

def test12() :
  fname="pdfs/textrank"
  pdf2txt(fname+".pdf")
  runWithFilter(fname+".txt",wk,sk,20,50,maybeWord)

def testx() :
  gm= GraphMaker()
  gm.digest('The cat sits near the cats on the mat.')
  for g in gm.gs :
     for e in g.triples() :
       print(e)
  for f,ft,r,t,tt in gm.edges() :
      #ppp('!!!!!!',f,ft,r,t,tt)
      print(f,'->',r,'->',t)
  pr=gm.pagerank()
  for w in pr.items() : print(w)
  showAllEdges(gm)

# interactive read, parse, show labeled edges loop
def test():
  gm= GraphMaker()
  while(True) :
    s = input('sentence>: ')
    if s=='' : break
    ppp(s)
    gm.digest(s)
    gm.graph()
    for g in gm.gs : 
      ppp(list(g.triples()))
      ppp('----')
    for f,ft,r,t,tt in gm.edges() :
      #ppp('!!!!!!',f,ft,r,t,tt)
      print(f,'->',r,'->',t)
    pr=gm.pagerank()
    for x in pr.items() : ppp(x)




  
#test() 

def go() :
  test1()
  test2()
  test3()
  test4()
  test5()
  test6()
  test7()
  test8()
  test9()
  test10()
  test11()

if __name__ == '__main__'  :
  pass
  #go()
  #test10()


