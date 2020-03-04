from nltk.parse.corenlp import CoreNLPDependencyParser

from .parser_api import *

# subclass using Stanford coreNLP

parserURL='http://localhost:9000'
class CoreNLP_API(NLP_API):
    def __init__(self, text):
        super().__init__(text)

        dparser = CoreNLPDependencyParser(url=parserURL)

        def parse(text) :
          return dparser.parse_text(text)

        # gss is a list of graph generators with
        # number of elements equal to the number of sentences

        chop = 2 ** 16
        gens = []

        while len(text) > chop:
          head = text[:chop]
          text = text[chop:]
          # ppp((head))
          if head:
            hs = list(parse(head))
            # ppp('PARSED')
            gens.append(hs)
        if gens:
          self.gss = [x for xs in gens for x in xs]
        else:
          self.gss = list(parse(text))


        #self.gss = list(dparser.parse_text(self.text))

        self.get_triples()
        self.get_lemmas()
        self.get_words()
        self.get_tags()

    def get_triples(self):
        if not self.triples:
            self.triples = []
            for gs in self.gss:
                self.triples.append(list(gs.triples()))
        return self.triples

    def _extract_key(gss, key):
        wss = []
        for gs in gss:
            ns = list(gs.nodes.items())
            ws = [None]*(len(ns)-1)
            for k, v in ns:
                #print("WORDDICT",v)
                ws[k-1] = v[key]
            wss.append(ws)
        return wss

    def get_lemmas(self):
        if not self.lemmas:
            self.lemmas = CoreNLP_API._extract_key(self.gss, 'lemma')
        return self.lemmas

    def get_words(self):
        if not self.words:
            self.words = CoreNLP_API._extract_key(self.gss, 'word')
        return self.words

    def get_tags(self):
      if not self.tags:
        self.tags = CoreNLP_API._extract_key(self.gss, 'tag')
      return self.tags

