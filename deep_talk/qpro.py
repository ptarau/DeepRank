from pyswip import *
from pyswip.easy import *
from graphviz import Digraph
import subprocess

from textcrafts import deepRank as dr
from textcrafts.sim import *
from textcrafts.vis import *
from textcrafts.parser_api import *


## PARAMS for Dialog Engines qpro.py and query.py

class talk_params(dr.craft_params):
    def __init__(self):
        super().__init__()
        self.corenlp = True
        self.quiet = True
        self.summarize = True
        self.quest_memory = 1
        self.max_answers = 3
        self.repeat_answers = 'yes'
        self.by_rank = 'no'
        self.personalize = 30
        self.show = True
        self.cloud = 36


params = talk_params()


####  start config aprameters ######

# finds absolute file name of Prolog companion qpro.pro
def pro():
    if __name__ == '__main__':
        # assume we just test this locally on its own
        return './qpro.pro'
    else:
        # assuming it is imported from a package and then run
        # to test this - possibly adapt after setup done
        f = __file__
        return f[:len(f) - 3] + ".pro"


####  end config aprameters ######


def say(what):
    print(what)
    if not params.quiet: subprocess.run(["say", what])


def go():
    # fNameNoSuf='examples/relativity'
    # fNameNoSuf='examples/bfr'
    fNameNoSuf = 'examples/const'
    print('dialog_about', fNameNoSuf)
    dialog_about(fNameNoSuf, None)


# generates Prolog facts dataset - depends on eval.py

def gen_pro_dataset():
    fd = ev.doc_dir
    fs = ev.all_doc_files
    for path in fs:
        fn = dr.justFname(path)
        prof = fd + "pro/" + fn
        txtf = dr.trimSuf(path)
        # print(txtf,'->',prof)
        print('GENERATING:', prof + ".pro")
        export_to_prolog(txtf, prof)


# interactive dialog
def talk_about(fNameNoSuf, params=params):
    return dialog_about(fNameNoSuf, None, params=params)


def dialog_about(fNameNoSuf, question, params=params):
    gm = export_to_prolog(fNameNoSuf, params=params)
    if params.show: gm.kshow(params.cloud, file_name="cloud.pdf", show=True)

    if params.summarize:
        print(gm)
    # for x in dr.best_edges(100,gm) : print('BEST EDGE',x)
    B = dr.best_line_graph(32, gm)
    gshow(B, attr="rel")
    prolog = Prolog()
    sink(prolog.query("consult('" + pro() + "')"))
    sink(prolog.query("load('" + fNameNoSuf + "')"))
    qgm = dr.GraphMaker(params=params)

    M = []
    log = []
    if isinstance(question, list):
        for q in question:
            say(q)
            process_quest(prolog, q, M, gm, qgm, fNameNoSuf, log, params=params)
    elif question:
        say(question)
        print('')
        dialog_step(prolog, question, gm, qgm, fNameNoSuf, log, params=params)
    else:
        while (True):
            question = input('?-- ')
            if not question: break
            process_quest(prolog, question, M, gm, qgm, fNameNoSuf, log, params=params)
    return process_log(log)


def process_log(log):
    l = len(log)
    qa_log = dict()
    for i in range(l):
        ls = []
        for a in log[i][1]:
            snum = a.split(':')[0]
            ls.append(int(snum))
        qa_log[i] = ls
    return qa_log


def process_quest(prolog, question, M, gm, qgm, fNameNoSuf, log, params=params):
    question = question + ' '
    if question in M:
        i = M.index(question)
        M.pop(i)
    M.append(question)
    if params.quest_memory > 0: M = M[-params.quest_memory:]
    Q = reversed(M)
    question = ''.join(Q)
    dialog_step(prolog, question, gm, qgm, fNameNoSuf, log, params=params)


# step in a dialog agent based given file and question
def dialog_step(prolog, question, gm, qgm, fNameNoSuf, log, params=params):
    query_to_prolog(question, gm, qgm, fNameNoSuf, params=params)
    rs = prolog.query("ask('" + fNameNoSuf + "'" + ",Key)")
    answers = [pair['Key'] for pair in rs]
    log.append((question, answers))
    if not answers:
        say("Sorry, I have no good answer to that.")
    else:
        for answer in answers:
            say(answer)
            print('')


def sink(generator):
    for _ in generator: pass


def getNERs(ws):
    from nltk.parse.corenlp import CoreNLPParser
    from textcrafts.corenlp_api import parserURL
    parser = CoreNLPParser(url=parserURL, tagtype='ner')
    ts = parser.tag(ws)
    for t in ts:
        if t[1] != 'O':
            yield t


# sends dependency triples to Prolog, as rececived from Parser
def triples_to_prolog(pref, qgm, f):
    ctr = 0
    for ts in qgm.triples():
        for x in ts:
            (fr, ft), r, (to, tt) = x
            print(pref + 'dep', end='', file=f)
            print((ctr, fr, ft, r, to, tt), end='', file=f)
            print('.', file=f)
        ctr += 1


def sents_to_prolog(pref, qgm, f):
    # s_ws_gen=dr.sent_words(qgm)
    s_ws_gen = enumerate(qgm.words())
    for s_ws in s_ws_gen:
        print(pref + 'sent', end='', file=f)
        print(s_ws, end='', file=f)
        print('.', file=f)


def ners_to_prolog(pref, qgm, f):
    if not params.corenlp:
        return
    # print("GENERATING NERS")
    s_ws_gen = enumerate(qgm.words())
    for s_ws in s_ws_gen:
        s, ws = s_ws
        ners = list(enumerate(getNERs(ws)))
        if ners:
            print(pref + 'ner', end='', file=f)
            print((s, ners), end='', file=f)
            print('.', file=f)


# sends summaries to Prolog
def sums_to_prolog(pref, k, qgm, f):
    if pref: return
    for sent in qgm.bestSentences(k):
        print('summary', end='', file=f)
        print(sent, end='', file=f)
        print('.', file=f)


# sends keyphrases to Prolog
def keys_to_prolog(pref, k, qgm, f):
    if pref: return
    for kw in qgm.bestWords(k):
        print(pref + "keyword('", end='', file=f)
        print(kw, end="')", file=f)
        print('.', file=f)


# sends edges of the graph to Prolog
def edges_to_prolog(pref, qgm, f):
    for ek in qgm.edgesInSent():
        e, k = ek
        a, aa, r, b, bb = e
        e = (k, a, aa, r, b, bb)
        print(pref + 'edge', end='', file=f)
        print(e, end='', file=f)
        print('.', file=f)
    if params.show and pref:
        dr.query_edges_to_dot(qgm)


# generic Prolog predicate maker
def facts_to_prolog(pref, name, facts, f):
    if isinstance(facts, dict):
        facts = facts.items()
    for fact in facts:
        print(pref + name, end='', file=f)
        print(fact, end='', file=f)
        print('.', file=f)
    print('', file=f)


# sends the computed ranks to Prolog
def ranks_to_prolog(pref, qgm, f):
    ranks = qgm.pagerank()
    facts_to_prolog(pref, 'rank', ranks, f)


# sends the words to lemmas table to Prolog
def w2l_to_prolog(pref, qgm, f):
    tuples = qgm.words2lemmas
    for r in tuples:
        print(pref + 'w2l', end='', file=f)
        print(r, end='', file=f)
        print('.', file=f)


# sends svo realtions to Prolog
def svo_to_prolog(pref, qgm, f):
    rs = qgm.bestSVOs(100)
    facts_to_prolog(pref, 'svo', rs, f)


# sends a similarity relation map to Prolog
def sims_to_prolog(pref, gm, qgm, f):
    # print(qgm.words)
    for qs in qgm.words2lemmas:
        qw, ql, qt = qs
        for cs in gm.words2lemmas:
            cw, cl, ct = cs
            if ql != cl:  # and qt==ct:
                if is_similar(ql, qt, cl, ct):
                    print('query_sim', end='', file=f)
                    print((ql, qt, cl, ct), end='', file=f)
                    print('.', file=f)


# sends a similarity relation map to Prolog
def rels_to_prolog(pref, gm, qgm, f):
    def sentId(touple):
        return touple[3]

    pr = gm.pagerank()
    ws = dict()
    rels = set()
    i = 0
    for w in pr:
        if dr.isWord(w):
            ws[w] = i
            i += 1
    for qs in qgm.words2lemmas:
        _, ql, qt = qs
        wn_tag = pos2tag(qt)
        if wn_tag != 'n': continue
        hypers = wn_hyper(3, ql, wn_tag)
        hypos = wn_hypo(3, ql, wn_tag)
        meros = wn_mero(3, ql, wn_tag)
        holos = wn_holo(3, ql, wn_tag)
        for h in hypers:
            if h in ws:
                rels.add((ql, 'is_a', h, -ws[h]))  # order in ranks = -ws[h]
        for h in hypos:
            if h in ws:
                rels.add((h, 'is_a', ql, -ws[h]))
        for h in meros:
            if h in ws:
                rels.add((h, 'part_of', ql, -ws[h]))
        for h in holos:
            if h in ws:
                rels.add((ql, 'part_of', h, -ws[h]))
    rels = sorted(rels, key=sentId, reverse=True)
    facts_to_prolog(pref, 'rel', rels, f)


# exporting to Prolog files needed to answer query

def export_to_prolog(fNameNoSuf, OutF=None, params=params):
    gm = dr.GraphMaker(params=params)
    gm.load(fNameNoSuf + '.txt')
    if not OutF:
        OutF = fNameNoSuf
    to_prolog('', gm, gm, OutF, params=params)
    return gm


def params_to_prolog(pref, f, params=params):
    rels = [('quest_memory', params.quest_memory),
            ('max_answers', params.max_answers),
            ('repeat_answers', params.repeat_answers),
            ('personalize', params.personalize),
            ('by_rank', params.by_rank)
            ]
    facts_to_prolog(pref, 'param', rels, f)


def personalize_for_query(gm, qgm, sk, wk):
    query_dict = dr.pers_dict(qgm)
    ranks = gm.rerank(query_dict)
    if params.show: gm.kshow(params.cloud, file_name="quest_cloud.pdf", show=True)

    def ranked(xs):
        # return xs
        for x in xs:
            yield x, ranks.get(x)

    # sents = [(w, r) for (w, r) in ranks if isinstance(w, int)]
    # words = [(w, r) for (w, r) in ranks if isinstance(w, str)]
    sents = ranked([x for (x, _) in gm.bestSentencesByRank(sk)])
    words = ranked(gm.bestWords(wk))
    # print('WORDS',words)
    return (sents, words)


# process a query and send it to Prolog
def query_to_prolog(text, gm, qgm, fNameNoSuf, params=params):
    qgm.digest(text)
    qfName = fNameNoSuf + '_query'
    to_prolog('query_', gm, qgm, qfName, params=params)


def personalized_to_prolog(pref, gm, qgm, personalize, f):
    count = personalize
    # print("COUNT",count)
    assert (isinstance(count, int))
    (sents, words) = personalize_for_query(gm, qgm, count, count)
    facts_to_prolog(pref, 'pers_sents', sents, f)
    facts_to_prolog(pref, 'pers_words', words, f)


# sends several fact predicates to Prolog
# small files are used that the pyswip activated Prolog will answer
# the pref='query_' marks file names with query_, while
# the empty prefix pref='' marks realtions describing a document
def to_prolog(pref, gm, qgm, fNameNoSuf, params=params):
    with open(fNameNoSuf + '.pro', 'w') as f:
        triples_to_prolog(pref, qgm, f)
        # print(' ',file=f)
        edges_to_prolog(pref, qgm, f)
        print(' ', file=f)
        ranks_to_prolog(pref, qgm, f)
        print(' ', file=f)
        w2l_to_prolog(pref, qgm, f)
        print(' ', file=f)
        sents_to_prolog(pref, qgm, f)
        print(' ', file=f)
        ners_to_prolog(pref, qgm, f)
        print(' ', file=f)
        svo_to_prolog(pref, qgm, f)
        print(' ', file=f)
        if pref:  # query only
            # sims_to_prolog(pref,gm,qgm,f)
            rels_to_prolog(pref, gm, qgm, f)  # should be after svo!
            if params.personalize > 0:
                personalized_to_prolog(pref, gm, qgm, params.personalize, f)
            params_to_prolog(pref, f, params=params)

        else:  # document only
            sums_to_prolog(pref, 10, qgm, f)
            print(' ', file=f)
            keys_to_prolog(pref, 10, qgm, f)
            print(' ', file=f)


# turns a sequence/generator into a file, one line per item yield
def seq2file(fname, seq):
    xs = map(str, seq)
    ys = interleave_with('\n', '\n', xs)
    text = ''.join(ys)
    string2file(fname, text)


# turns a file into a (string) generator yielding each of its lines
def file2seq(fname):
    with open(fname, 'r') as f:
        for l in f: yield l.strip()


# turns a string into given file
def string2file(fname, text):
    with open(fname, 'w') as f:
        f.write(text)


# turns content of file into a string
def file2string(fname):
    with open(fname, 'r') as f:
        s = f.read()
        return s.replace('-', ' ')


# interleaves list with separator
def interleave(sep, xs):
    return interleave_with(sep, None, xs)


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


def chat(FNameNoSuf):
    return dialog_about('examples/' + FNameNoSuf, None)


def pdf_chat(FNameNoSuf, params=params):
    return pdf_chat_with("pdfs", FNameNoSuf, params=params)


def pdf_chat_with(Folder, FNameNoSuf, about=None, params=params):
    fname = Folder + "/" + FNameNoSuf
    dr.pdf2txt(fname + ".pdf")
    return dialog_about(fname, about, params=params)


def pdf_quest(Folder, FNameNoSuf, QuestFileNoSuf, params=params):
    Q = []
    qfname = Folder + "/" + QuestFileNoSuf + ".txt"
    qs = list(file2seq(qfname))
    return pdf_chat_with(Folder, FNameNoSuf, about=qs, params=params)


def txt_quest(Folder, FNameNoSuf, QuestFileNoSuf, params=params):
    Q = []
    qfname = Folder + "/" + QuestFileNoSuf + ".txt"
    qs = list(file2seq(qfname))
    # print('qs',qs)
    return dialog_about(Folder + "/" + FNameNoSuf, qs, params=params)


ppp = print

# all_ts()

print('prolog companion', pro())

if __name__ == '__main__':
    pass
