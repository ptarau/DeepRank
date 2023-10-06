from nltk.corpus import wordnet as wn

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def is_similar(u, Pu, v, Pv):
    pu = pos2tag(Pu)
    pv = pos2tag(Pv)
    if pu and pv:
        mysim = sim1(u, pu, v, pv)
        wup = sim2(u, pu, v, pv)
        avg = (wup + mysim) / 2
        if avg > 0.7:
            return True
        else:
            return False


def sim2(u, pu, v, pv):
    m = 0
    for i in wn.synsets(u, pu):
        for j in wn.synsets(v, pv):
            s = i.wup_similarity(j)
            if s: m = max(m, s)
    return m


def sim1(u, pu, v, pv):
    us = set(wn.synsets(u, pu))
    if not us:
        u = wn.morphy(u, pu)
        if u:
            us = set(wn.synsets(u, pu))
    vs = set(wn.synsets(v, pv))
    if not vs:
        v = wn.morphy(v, pv)
        if v:
            vs = set(wn.synsets(v, pv))
    hus = set()
    for x in us: hus = hus.union(set(x.hypernyms()))
    for x in us: hus = hus.union(set(x.hyponyms()))
    hvs = set()
    for x in vs: hvs = hvs.union(set(x.hypernyms()))
    # for x in vs : hvs=hvs.union(set(x.hyponyms()))
    us = us.union(hus)
    vs = vs.union(hvs)
    cs = us.intersection(vs)
    if cs:
        return sigmoid(len(cs))
    else:
        return 0


def pos2tag(pos):
    if not pos:
        return None
    c = pos[0]
    if c is 'N':
        return 'n'
    elif c is 'V':
        return 'v'
    elif c is 'J':
        return 'a'
    elif c is 'R':
        return 'r'
    else:
        return None


# basic wordnet relations

def wn_hyper(k, w, t): return wn_rel(hypers, 2, k, w, t)


def wn_hypo(k, w, t): return wn_rel(hypos, 2, k, w, t)


def wn_mero(k, w, t): return wn_rel(meros, 2, k, w, t)


def wn_holo(k, w, t): return wn_rel(holos, 2, k, w, t)


def wn_syn(k, w, t): return wn_rel(id, 2, k, w, t)


def id(w): return [w]


def hypos(s): return s.hyponyms()


def hypers(s): return s.hypernyms()


def meros(s): return s.part_meronyms()


def holos(s): return s.part_holonyms()


#  ADJ,ADJ_SAT, ADV, NOUN, VERB = 'a','s', 'r', 'n', 'v'

def wn_tag(T):
    c = T[0].lower()
    if c in 'nvr':
        return c
    elif c == 'j':
        return 'a'
    else:
        return None


def wn_rel(f, n, k, w, t):
    related = set()
    for i, syns in enumerate(wn.synsets(w, pos=t)):
        if i >= n: break
        for j, syn in enumerate(f(syns)):
            if j >= n: break
            # print('!!!!!',syn)
            for l in syn.lemmas():
                # print('  ',l)
                s = l.name()
                if w == s: continue
                s = s.replace('_', ' ')
                related.add(s)
                if len(related) >= k: return related
    return related


def simtest():
    w, tag = 'car', 'n'
    print(wn_rel(id, 2, 300, w, tag))
    print(wn_rel(hypers, 2, 300, w, tag))
    print(wn_rel(meros, 2, 300, w, tag))
    print(wn_rel(holos, 2, 300, w, tag))
    print('')
    print(wn_syn(5, w, tag))

######
