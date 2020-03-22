import networkx as nx
import math
from collections import defaultdict
import subprocess
from inspect import getframeinfo, stack
from .down import ensure_nlk_downloads
from nltk.corpus import words
from nltk.corpus import stopwords
from graphviz import Digraph
from .sim import *
from .vis import *

class craft_params:
  def __init__(self) :
    # Graph building, parsing ranking

    self.corenlp=True

    self.abstractive='no'
    self.show=True # graph pics?

    # for LINKS, RANKING, SUMMARIES AND KEYPHRASES
    self.all_recs  = True  # sentence recommendatations
    self.giant_comp = False # only extract from giant comp
    self.noun_defs = True
    self.noun_self = False

    self.sent_count=5
    self.word_count=6
    self.rel_count=10
    self.dot_count=16

   # formula for adjusting rank of long or short sentences
  def adjust_rank(self,rank,length,avg) :
    #adjust = 1 + math.sqrt(1 + abs(length - avg))
    adjust = 1 + math.log(1+abs(length-avg))
    newr=rank/adjust
    #print('ADJUST',adjust,length,avg)
    return newr

  # choice of toolkit
  def get_toolkit(self) :
    if self.corenlp:
      from  .corenlp_api import CoreNLP_API
      toolkit=CoreNLP_API
    else:
      from .stanfordnlp_api import StanTorch_API
      toolkit=StanTorch_API
    return toolkit

params=craft_params()

print('RUNNING textcrafts',__file__)

def ppp(*args) :
  caller = getframeinfo(stack()[1][0])

  print('DEBUG:',
        caller.filename.split('/')[-1],
        '->',caller.lineno,end=': ')
  print(*args)

print("TOOLKIT iS corenlp: ",params.corenlp)

def make_word_dict(fname):
    wd = set(words.words())

    def add_from_file(fname):
        added = True
        try:
            with open(fname, 'r') as f:
                for w in f.readlines():
                    wd.add(w[:-1])
                    #if w.startswith('space') : print(w[:-1],len(w))
        except:
            added = False
        return added
    if add_from_file('words.txt'):
       print('Added words.txt')
    if add_from_file(fname):
        print('Domain-specific dictionary', fname, 'added.')
    return wd


word_dict = make_word_dict('kb.txt')

stop_words = set(stopwords.words('english'))

# recognizers for for several word types


def isString(w):
    return isinstance(w, str)


def isWord(w):
    if not isString(w):
        return False
    # if isStopWord(w) : return False
    wl = len(w)
    return wl > 1 and (isName(w) or inDict(w))


def isName(w):
    return w.isalnum() and w[0].isupper()


def isStopWord(w):
    return w.lower() in stop_words


def maybeWord(w):
    if not isString(w):
        return False
    clear_word = isWord(w) and len(w) > 1
    return clear_word or isSpec(w) or isPunct(w) or w.isdigit() or hasDash(w)


def isSpec(w):
    return w in ['a', 'A', "'s", 'I'] or (w[0].isupper() and w.endswith('.'))


def isPunct(w):
    return w in ",.?;:-'()" or w == '"'


def hasDash(w):
    ws = w.split('-')
    if len(ws) < 2:
        return False
    for w in ws:
        if '' == w or not maybeWord(w):
            return False
    return True


def inDict(w):
    def ends_well(w):
        if w.endswith('ing'):
            return True
        if w.endswith('ed'):
            return True
        if w.endswith('ility'):
            return True
        if w.endswith('ly'):
            return True
        if w.endswith('er'):
            return True
        if w.endswith('st'):
            return True
        return False

    if w in word_dict:
        return True
    if w.capitalize() in word_dict:
        return True
    if w.lower() in word_dict:
        return True
    if ends_well(w):
        return True
    if 's' == w[-1]:
        v = w[:-1]
        return ends_well(v)
    return False

# ensures the sentence does not have
# strings that do not make sense as keywords
# or words in a sentence


def isCleanSent(sent):
    if len(sent) < 3 : return False
    ok = True
    for w in sent:
        if not w :
          return False
        if maybeWord(w):
            continue
        if inDict(w):
            continue
        ok = False
        # ppp('UNCLEAN',w)
        break
    return ok

# recognize nouns, from their POS tags


def isNoun(tag):
    # ppp('TAG=',tag)
    return tag[0] == 'N'

# recognize adjectives


def isAdj(tag):
    return tag[0] == 'J'

# recognize verbs, from their POS tahs


def isVerb(tag):
    return tag[0] == 'V'

# recognize subjects, from a relation label


def isSubj(rel):
    return rel.find('subj') >= 0

# recognize obbjects, from a relation label


def isObj(rel):
    # return rel=='obj' or rel == 'dobj' or rel=='iobj'
    return rel.find('obj') >= 0


def isMod(rel):
    return rel == 'nmod' or rel == 'amod'

# sentece idetifiers are natural numbers


def isSent(s):
    return isinstance(s, int)

# true for any argument


def isAny(x):
    return True

# class, building and ranking a text graph from a document or string

class GraphMaker:
    def __init__(self,params=params,file_name=None,text=None,corenlp=False):
        print("PARAMETER CLASS:",type(params).__name__,'\n')
        self.params=params
        self.api_classname = params.get_toolkit()
        #self.api_classname = api_classname
        self.clear()
        if file_name :
          self.load(file_name)
        elif text :
          self.digest(text)
        #else :
          #print('*** text OR file_name optional parameter missing')

    # # clear saved state
    def clear(self):
        self.maxcc = None
        self.gs = None
        self.nxgraph = None
        self.ranked = None
        self.words2lemmas = set()
        self.noun_set = dict()
        self.svo_edges_in_graph = []

    def triples(self):
       return self.gs[0]

    def lemmas(self):
      return self.gs[1]

    def words(self):
        return self.gs[2]

    def tags(self):
      return self.gs[3]

    # digest a file
    def load(self, fname):
        with open(fname, 'r') as f:
            text = f.read()
        self.digest(text)

    # digest a string using dependecy parser
    def digest(self, text):
        self.clear()
        self.gs = self.api_classname(text).get_all()
 

    # sentence as sequence of words generator
    def sentence(self):
        for ws in self.words() :
            yield str.join(' ', ws)


    # curates, reverses and adds some new edges
    # yields an <edge, sentence in which it occurs> pair
    def edgesInSent(self):
        self.svo_edges_in_graph = []

        def noun_to_def(x, tx, k):
            if self.params.noun_defs:
                k_ = self.noun_set.get(x)
                if k == k_:
                    yield (x, tx, 'first_in', k, 'SENT')

        def edgeOf(k):
            d = w2l(self.words(),self.lemmas(),self.tags(),k)
            merge_dict(self.words2lemmas, d)
            make_noun_set(self.noun_set,self.lemmas(),self.tags(),k)
            svo_edges_in_sent = []
            for triple in self.triples()[k]:
                #ppp('TRIPLE',triple)
                fr, rel, to = triple
                lfrom, ftag = d[fr[0]]
                lto, ttag = d[to[0]]

                # vn is True it is an s->v or o->v link
                so = isSubj(rel) or isObj(rel)
                vn = isVerb(ftag) and isNoun(ttag) and so
                if rel == 'punct' and ttag == '.':
                    # sentence points to predicate verb
                    yield (k, 'SENT', 'predicate', lfrom, ftag)
                #elif rel == 'punct' : continue
                elif vn:
                    # collects vs and vo links to merge them later into svo
                    if lfrom != lto :
                      svo_edges_in_sent.append((lfrom, ftag, rel, lto, ttag))
                    yield lfrom, ftag, rel, lto, ttag  # verb to noun
                    yield k, 'SENT', 'about', lto, ttag  # sent to noun
                    # all words recommend sentence
                    # yield lfrom,ftag,'recommends',k,'SENT' # verb to sent - in elif !
                    for e in noun_to_def(lto, ttag, k,):
                        yield e  # noun to sent
                    if self.params.noun_self:
                        yield lto, ttag, 'self', lto, ttag
                elif isNoun(ttag):  # e.g. nmod relation
                    # ppp('x-->n',k,lfrom,ftag,rel,lto,ttag)
                    yield lfrom, ftag, rel, lto, ttag
                    for e in noun_to_def(lto, ttag, k,):
                        yield e  # noun to sent
                    if self.params.noun_self:
                        yield lto, ttag, 'self', lto, ttag
                    # yield lfrom, ftag, 'recommends', k, 'SENT' # dependent of noun to sent
                else:  # yield link as is
                    yield lto, ttag, rel, lfrom, ftag
                    # all words recommend sentence
                    if self.params.all_recs:
                        yield lto, ttag, 'recommends', k, 'SENT'

                # merge compound terms, make their parts recommend them
                if isNoun(ftag) and isNoun(ttag) and rel == 'compound':
                    comp = lto + ' ' + lfrom
                    yield lfrom, ftag, 'fused', comp, ftag
                    yield lto, ttag, 'fused', comp, ttag
                    for e in noun_to_def(comp, ttag, k):
                        yield e
                    if self.params.noun_self:
                        yield comp, ttag, 'self', comp, ttag
            # collect svo relations
            self.svo_edges_in_graph.append(to_svo(k, svo_edges_in_sent))

        for k in range(len(self.triples())):
            for e in edgeOf(k):
                # collects words at the two ends of e
                # self.addWordsIn(e)
                yield e, k

    # yields  the edge. possibly for each sentence where is found
    def multi_edges(self):
        for e, k in self.edgesInSent():
            yield e

    def edges(self):
        for e in set(self.multi_edges()):
            yield e

    # returns final networkx text graph
    def graph(self):
        if(self.nxgraph):
            return self.nxgraph
        dg = nx.DiGraph()

        for e in self.edges():
            f, tf, r, t, tt = e
            dg.add_edge(f, t, rel=r)

        self.nxgraph = dg
        # ppp('DG:',dg,'END')

        # ppp('NOUN_SET',self.noun_set)
        return dg

    def size(self):
        return (self.nxgraph.number_of_nodes(), self.nxgraph.number_of_edges())

    # ranks (unless ranked and stored as such) the text graph
    def pagerank(self, pers=None):
        if self.ranked:
            return self.ranked
        g = self.graph()

        pr = self.runPagerank(g, pers)
        self.ranked = pr
        if not self.params.giant_comp:
            return pr
        ccs = list(nx.strongly_connected_components(g))
        lc = len(ccs)
        #ppp('LEN_COMPS', lc)
        if lc < 4:
            self.maxcc = max(ccs, key=len)
        return pr

    def rerank(self, pers=None):
        self.ranked = None
        # ppp("RERANK",len(pers))
        return self.pagerank(pers=pers)

    # extracts best k nodes passing filtering test
    def bestNodes(self, k, filter):
        # g=self.graph()
        # comps=list(nx.strongly_connected_components(g))

        pr = self.pagerank()
        i = 0
        ns = []  # not a set - that looses order !!!
        for x, r in pr.items():
            if i >= k:
                break
            # ppp('RANKED',x,r)
            if filter(x):
                # ppp('FILTERED',x,r,'MC')
                if not self.maxcc or x in self.maxcc:
                    if not x in ns:
                        ns.append(x)
                        i += 1
        return ns

    # specialization returning all best k nodes
    def bestAny(self, k):
        return self.bestNodes(k, lambda x: True)

    # specialization returning best k sentence nodes
    def bestSentencesByRank(self, k, filter=isAny):
        best = self.bestNodes(100+k, isSent)
        if not best:
            return
        #ppp('BEST SENTS:',best)
        c = 0
        for i in best:

            if not filter(i):
                continue
            #g = self.gs[i]
            #ts,lems,ws,ps=g
            # ppp('LEMS',lems)
            lems=self.lemmas()[i]
            if isCleanSent(lems):
                sent = self.words()[i]
                #sent=str.join(' ',list(gwords(g)))
                yield (i, sent)
                c += 1
            # else : ppp('SENT UNCLEAN',i)
            if c >= k:
                break

    def bestSentences0(self, k, filter=isAny):
        for i_ws in sorted(self.bestSentencesByRank(k, filter=filter)):
            yield i_ws


    def bestSentences(self, k, filter=isAny):
        xs = self.bestSentences0(k, filter=isAny)
        if self.params.abstractive == 'yes':
            xs = list(xs)
            pr = self.pagerank()  # make pr an (ordered) dict
            wg = nx.DiGraph()
            fs = set()
            ts = set()
            for s_ws in xs:
                _, ws = s_ws
                if not ws or len(ws) < 5:
                    continue
                f = ws[0]
                t = ws[-1]
                if not f or not t:
                    continue
                fs.add(f)
                ts.add(t)

                #word_graph(wg, pr, s_ws)
                # print('LLLL',wg.number_of_edges())
            #gshow1(wg)
            for f in fs:
                if f not in wg.nodes():
                    continue
                for t in ts:
                    if t not in wg. nodes():
                        continue
                    ps = nx.dijkstra_path(wg, f, t)
                    if ps:
                        print("DPATH", ps)

        for s_ws in xs:
            yield s_ws



    # specialization returning best k word nodes
    def bestWords(self, k):
        # ppp('NOUNS',self.noun_set)
        c = 0
        best = self.bestNodes(100+k, maybeWord)
        #ppp('BEST WORDS:',best)
        for w in best:
            if c >= k:
                break
            #ppp('GOOD WORD', w)
            if not isStopWord(w) and self.hasNoun(w):
                yield(w)
                #ppp('BWORD',w)
            c += 1

    def kshow(self,k,file_name="cloud.pdf",show=params.show):
      d=dict()
      for w in self.bestWords(k) :
        d[w]=self.pagerank()[w]
      show_ranks(d,file_name=file_name,show=show)

    # true if a phrase has a noun in it
    def hasNoun(self, w):
        ws = w.split(' ')
        for v in ws:
            if v in self.noun_set:
                return True
        return False

    # runs PageRank on text graph
    def runPagerank(self, g, pers):
        #if pers : ppp('PERSONALIZED',pers)
        d = nx.pagerank(g, personalization=pers)
        # ppp("PR",d)

        # normalize sentence ranks by favoring those close to average rank
        sents = self.words()
        lens = list(map(len, sents))
        #ppp('LENS:', lens)
        avg = sum(lens) / len(lens)

        #ppp('AVG SENT LENGTH:', avg)

        # reranks long sentences
        i = 0
        for ws in sents:  # this makes i going over sentence ids
            # ppp('WS:',ws)
            if i in d:
                l = len(ws)
                r = d[i]
                newr = self.params.adjust_rank(r, l, avg)
                d[i] = newr
                #if l<6 : ppp(r,'--->',newr,l,'ws=',ws)
                i += 1

        sd = sorted(d, key=d.get, reverse=True)

        return dict((k, d[k]) for k in sd)

    # extracts k highest ranked SVO triplets
    def bestSVOs(self, k):
        rank_dict = self.pagerank()
        # ppp('PRANK',rank_list)
        ranked = []  # should not be a set !
        for rs in self.svo_edges_in_graph:
            for r in rs:
                # ppp('SVO',r)
                (f, _), (rel, _), (t, _), sent_id = r
                srank = rank_dict[f]
                orank = rank_dict[t]
                if srank and orank:
                    sorank = (2*srank+orank)/3
                    ranked.append((sorank, (f, rel, t, sent_id)))
        ranked = sorted(ranked, reverse=True)
        i = 0
        exts = set()
        seen = set()
        for (_, e) in ranked:
            i += 1
            if i > k:
                break
            # ppp('SVO_EDGE',e)
            if e in seen:
                continue
            seen.add(e)
            yield e
            for xe in self.extend_with_wn_links(e, rank_dict):
                f, _, t, _ = xe
                if wn.morphy(f.lower()) != wn.morphy(t.lower()):
                    exts.add(xe)
        i = 0
        for xe in exts:
            i += 1
            if i > k:
                break
            # ppp('XE',xe)
            yield xe

    # adds wordnet-derived links to a dictionary d
    # we tag them with is_a or part_of
    def extend_with_wn_links(self, e, d):
        s, v, o, sent_id = e
        m = 1  # how many of each are taken
        for x in wn_holo(m, s, 'n'):
            if x in d:
                yield (s, 'part_of', x, sent_id)
        for x in wn_mero(m, s, 'n'):
            if x in d:
                yield (x, 'part_of', s, sent_id)
        for x in wn_hyper(m, s, 'n'):
            if x in d:
                yield (s, 'is_a', x, sent_id)
        for x in wn_hypo(m, s, 'n'):
            if x in d:
                yield (x, 'is_a', s, sent_id)
        for x in wn_holo(m, o, 'n'):
            if x in d:
                yield (o, 'part_of', x, sent_id)
        for x in wn_mero(m, o, 'n'):
            if x in d:
                yield (x, 'part_of', o, sent_id)
        for x in wn_hyper(m, o, 'n'):
            if x in d:
                yield (o, 'is_a', x, sent_id)
        for x in wn_hypo(m, o, 'n'):
            if x in d:
                yield (x, 'is_a', o, sent_id)

    # visualize filtered set of edges with graphviz
    def toDot(self, k, filter, svo=False,
              show=params.show, fname='textgraph.gv') :
        if not show : return
        dot = Digraph()
        g = self.graph()
        best = self.bestNodes(k, filter)
        for f, t in g.edges():
            if f in best and t in best:
                dot.edge(str(f), str(t))
        if svo:
            svos = set()
            for (s, v, o, _) in self.bestSVOs(k):
                svos.add((s, v, o))
            for e in svos:
                s, v, o = e
                dot.edge(s, o, label=v)
        showGraph(dot, show=show, file_name=fname)

    # visualize filtered set of edges as graphviz dot graph
    def svoToDot(self, k):
        dot = Digraph()
        for e in self.bestSVOs(3*k):
            s, v, o = e
            dot.edge(s, o, label=v)
        showGraph(dot)

    # specialize dot graph to words
    def wordsToDot(self, k):
        self.toDot(k, isWord)

    # specialize dot senteces graph words
    def sentsToDot(self, k):
        self.toDot(k, isSent)

    # visualize mixed sentence - word graph
    def allToDot(self, k):
        self.toDot(k, lambda x: True)

    def keyphrases(self):
        L = ['--- KEYPHRASES ---']
        for w in self.bestWords(params.word_count):
            L.append(w + ';')
        return '\n'.join(L)

    def summary(self):
        L = ['--- SUMMARY ---']
        for s in self.bestSentences(params.sent_count):
            n, ws = s
            L.append(f'{n} : ' + ''.join([' '+w for w in ws]))
        return '\n'.join(L)

    def relations(self) :
        L = ['--- RELATIONS ---']
        L.extend(list(map(str, self.bestSVOs(params.rel_count))))
        return '\n'.join(L)

    def __repr__(self):
        s = []
        s.append('--- GraphMaker object ---')
        s.append(f'nodes: {self.graph().number_of_nodes()}')
        s.append(f'edges: {self.graph().number_of_edges()}')
        s.append(self.keyphrases())
        s.append(self.summary())
        s.append(self.relations())
        s.append(80*'-')
        return '\n'.join(s)


def fix_par(w):
    u = w.upper()
    if u == '-LRB-':
        return '('
    elif u == '-RRB-':
        return ')'
    elif u == '-LSB-':
        return '['
    elif u == '-RSB-':
        return ']'
    else:
        return w

def pers_dict(qgm):
    return dict(
        (w, r) for (w, r) in qgm.pagerank().items()
        if maybeWord(w) and not isStopWord(w)
    )


# returns a dict of lemmas for word nodes in in g
def w2l(wss,lss,pss,k):
    d = dict()
    ln = len(wss[k])
    for i in range(ln) :
        w = wss[k][i]
        l = lss[k][i]
        p =  pss[k][i]
        if(w):
            d[w] = (l, p)
    return d

# adds to given dict d nouns in nodes of g


def make_noun_set(d,lss,pss, k):
    ln = len(lss[k])
    for i in range(ln):
      l = lss[k][i]
      p = pss[k][i]
      if(l) and isNoun(p):
        if not l in d:
          d[l] = k


def merge_dict(tuples, d):
    for k in d:
        l, lt = d[k]
        # ppp(l,lt)
        tuples.add((k, l, lt))

# turns into an svo triple two s and o links a verb points to


def to_svo(k, rs):
    #ppp('TRYING SVO',k,rs)
    s = dict()
    o = dict()
    svo = []
    for r in rs:
        f, ft, rel, t, tt = r
        if not (isWord(f) and isWord(t)):
            continue
        if isSubj(rel):
            s[(f, ft)] = (t, tt)
        elif isObj(rel):
            o[(f, ft)] = (t, tt)
        # else : ppp(k,'*** UNEXPCTED REL',f,t,rel)
    for vt in s:
        if vt in o:
            svo.append((s[vt], vt, o[vt], k))  # k=sent where found
    #for x in svo : ppp(k,'GOT SVO',x)
    return svo

# the walk should go along dependency edges !!!
def word_graph(g, pr, s_ws):
      pr = dict(pr)
      s, ws0 = s_ws
      ws = [w for w in ws0 if not isStopWord(w)]
      l = len(ws)
      for i in range(0, l - 1):
        f = ws[i]
        t = ws[i + 1]
        r1 = pr.get(f)
        if not r1: r1 = pr.get(f.lower())
        r2 = pr.get(t)
        if not r2: r2 = pr.get(f.lower())
        if not r1: r1 = 0
        if not r2: r2 = 0
        r = 0
        if not r1:
          r = r2
        elif not r2:
          r = r1
        else:
          r = (r1 + r2) / 2  # we will be looking for shortest
        if r == 0:
          r = float('inf')
        else:
          r = math.exp(1 - r)
        g.add_edge(f, t, weight=r)
        # print("EDGE",(f,t,r))

def best_edges(k,g) :
  G=g.graph()
  pr = nx.pagerank(G)
  lg=nx.line_graph(G)
  #pr=nx.pagerank(G)
  prl=list(nx.pagerank(lg,personalization=None).items())
  #for p in prl : print(p)
  wes=[(e,r) for  (e,r) in prl if isinstance(e[0],str) and isinstance(e[1],str)]
  wes.sort(key=lambda x : x[1], reverse=True)
  xs=map(lambda x : x[0],wes)
  yield from take(k,xs)

def best_line_graph(k,g) :
  B=nx.DiGraph()
  for x,y in best_edges(k,g) :
    r=g.graph()[x][y]["rel"]
    #print('REL',x,y,r)
    B.add_edge(x,y)
    if r: B[x][y]["rel"]=r
  return B

# shows summaries and keywords from file
# extracts highest ranked svo relations  and visualizes
# highest ranked filtered word to word edges as dot graph
# if svo optional arg is set to True, adns svo links to the graph
def runWithFilter(fileName,filter=(lambda x: True)):
  gm = GraphMaker(file_name=fileName)
  print(gm)
  dotName = trimSuf(fileName) + ".gv"
  gm.toDot(params.dot_count, filter, svo=True, fname=dotName, show=params.show)
  return gm

def print_summary(xs):
    for s in xs:
      n, ws = s
      print(n, ':', end='')
      for w in ws:
        print(' ', w, end='')
      print('')
    print('')

def showAllEdges(g, file_name='textgraph.gv'):
    dot = Digraph()
    for e in g.edges():
        # ppp('EDGE:',e)
        f, ft, rel, t, tt = e
        rl = ft+'_'+str(rel)+'_'+tt
        dot.edge(str(f), str(t), label=rl)
    showGraph(dot, file_name=file_name)


def query_edges_to_dot(qgm):
    showAllEdges(qgm, file_name='query_graph.gv')

    # displays textgraph with graphviz

def showGraph(dot, show=False, file_name='textgraph.gv'):
    dot.render(file_name, view=show)

def gshow1(g, **kwargs):
    dot = Digraph()
    for e in g.edges():
        # print('EEEE',e)
        f, t = e
        w = g[f][t]['weight']
        dot.edge(str(f), str(t), label=str(w))
    showGraph(dot)


# extracts file name from path
def path2fname(path):
    return path.split('/')[-1]


def trimSuf(path):
    return ''.join(path.split('.')[:-1])


def justFname(path):
    return trimSuf(path2fname(path))


def pdf2txt(fname):
    subprocess.run(["pdftotext", fname])


def take(k, seq):
    c = 0
    for x in seq:
        if c >= k:
            return
        yield x
        c += 1

def my_path() :
  if __name__=='__main__' :
    return "./"
  else :
    return __file__


#print('deepRank:',my_path())
