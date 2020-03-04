import stanfordnlp
from .parser_api import *
import os
import sys

# subclass using  torch-based  stanfordnlp - Apache licensed
class StanTorch_API(NLP_API):

    def start_pipeline():
        mfile = os.getenv("HOME") + \
            '/stanfordnlp_resources/en_ewt_models'
        if not os.path.exists(mfile):
          stanfordnlp.download('en', confirm_if_exists=True, force=True)
        sout = sys.stdout
        serr = sys.stderr
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f
        # turn output off - too noisy
        nlp = stanfordnlp.Pipeline()
        sys.stdout = sout
        sys.stderr = serr
        # turn output on again
        return nlp

    nlp = start_pipeline()

    def __init__(self, text):
        super().__init__(text)
        self.doc = self.start_parser(text)

    def get_triples(self):
        if not self.triples:
            tss = []
            for s in self.doc.sentences:
                ts = []
                for dep_edge in s.dependencies:
                    if dep_edge[1]=='root' :
                      continue # compatibility with coreNLP
                    source = (dep_edge[0].text, dep_edge[0].pos)
                    target = (dep_edge[2].text, dep_edge[2].pos)
                    t = (source,  dep_edge[1], target)
                    #print('DEPEDGE', len(dep_edge),t)
                    if None in t:
                      print("BAD EDGE DATA", t)
                      continue
                    ts.append(t)
                tss.append(ts)
                self.tuples = tss
        return self.tuples

    def get_words_lemmas_tags(self):
        if not self.lemmas or not self.words:
            wss = []
            lss = []
            pss = []
            for s in self.doc.sentences:
                ws = []
                ls = []
                ps = []
                for w in s.words:
                    ws.append(w.text)
                    l=w.lemma
                    if not l :
                      l=w.text
                    ls.append(l)
                    ps.append(w.xpos)
                if None in ws or None in ls or None in ps:
                  print("BAD DATA",ls) # when None lemmas returned by parser
                  # continue
                wss.append(ws)
                lss.append(ls)
                pss.append(ps)
            self.words = wss
            self.lemmas = lss
            self.tags=pss

    def get_words(self):
        self.get_words_lemmas_tags()
        return self.words

    def get_lemmas(self):
        self.get_words_lemmas_tags()
        return self.lemmas

    def get_tags(self):
        self.get_words_lemmas_tags()
        return self.tags

    def start_parser(self, text):
        sout = sys.stdout
        serr = sys.stderr
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f
        # turn output off - too noisy
        self.dparser = StanTorch_API.nlp(text)
        sys.stdout = sout
        sys.stderr = serr
        # turn output on again
        return self.dparser
