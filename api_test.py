import os
import sys
from textcrafts.deepRank import maybeWord, isAny, pdf2txt
from textcrafts import GraphMaker

from textcrafts.corenlp_api import *
from textcrafts.stanfordnlp_api import *

from timeit import timeit as tm

def apply_api(api, fname):
    with open(fname, 'r') as f:
        ls = f.readlines()
        text = "".join(ls)
        return api(text).get_all()


def t1():
    print('with coreNLP')
    print('')
    text = 'The happy cat sleeps. The dog just barks today.'
    p = CoreNLP_API(text)
    print(p.get_triples())
    print('')
    print(p.get_lemmas())
    print('')
    print(p.get_words())
    print('')
    print(p.get_triples())
    print('-'*50)
    print('')


def t2():
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
    print(p.get_tags())
    print('')

# benchmark


def bm1(fname):
    api = CoreNLP_API
    (ds, ls, ws, _) = apply_api(api, fname)
    print('coreNLP', 'sents=', len(ws))


def bm2(fname):
    api = StanTorch_API
    (ds, ls, ws, _) = apply_api(api, fname)
    print('stanfordnlp', 'sents=', len(ws))


def bm():

    fname = 'examples/const.txt'
    #fname = 'examples/einstein.txt'
    #fname = 'examples/tesla.txt'
    for _ in range(3):
        print(fname,tm(lambda: bm1(fname), number=1))
        print(fname,tm(lambda: bm2(fname), number=1))

if __name__ == '__main__':
    t1()
    t2()
    #bm()
