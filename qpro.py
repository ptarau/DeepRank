from nltk.parse.corenlp import CoreNLPParser

from pyswip import *
from pyswip.easy import *
from graphviz import Digraph
from sim import *
import deepRank as dr
import subprocess
from sim import *
import eval as ev
from params import *


def say(what) :
  print(what)
  if not quiet : subprocess.run(["say", what])

def go() :
  #fNameNoSuf='examples/relativity'
  #fNameNoSuf='examples/bfr'
  fNameNoSuf='examples/const'
  print('dialog_about',fNameNoSuf)
  dialog_about(fNameNoSuf,None)

# generates Prolog facts dataset

def gen_pro_dataset() :
  fd=ev.doc_dir
  fs=ev.all_doc_files
  for path in fs :
    fn=dr.justFname(path)
    prof=fd+"pro/"+fn
    txtf=dr.trimSuf(path)
    #print(txtf,'->',prof)
    print('GENERATING:',prof+".pro")
    export_to_prolog(txtf,prof)

# interactive dialog
def talk_about(fNameNoSuf):
  return dialog_about(fNameNoSuf,None)

def dialog_about(fNameNoSuf,question) :
  gm=export_to_prolog(fNameNoSuf)

  if summarize :
    wk,vk,sk = 6,6,3
    dr.print_keys(gm.bestWords(wk))
    dr.print_rels(gm.bestSVOs(vk))
    print('SUMMARY')
    dr.print_summary(gm.bestSentences(sk))

  prolog = Prolog()
  sink(prolog.query("consult('qpro.pro')"))
  sink(prolog.query("load('"+fNameNoSuf+"')"))
  qgm=dr.GraphMaker()

  M = []
  log=[]
  if isinstance(question,list):
    for q in question :
      say(q)
      process_quest(prolog,q, M, gm, qgm, fNameNoSuf,log)
  elif question :
    say(question)
    print('')
    dialog_step(prolog,question,gm,qgm,fNameNoSuf,log)
  else :
    while(True) :
      question=input('?-- ')
      if not question : break
      process_quest(prolog,question, M, gm, qgm, fNameNoSuf,log)
  return process_log(log)

def process_log(log) :
  #print('LOG:', log)
  #with open(logfile,'w') as lf :
  l=len(log)
  qa_log=dict()
  for i in range(l) :
    ls=[]
    for a in log[i][1] :
      snum=a.split(':')[0]
      ls.append(int(snum))
    qa_log[i]=ls
  return qa_log

def process_quest(prolog,question,M,gm,qgm,fNameNoSuf,log) :
  question = question + ' '
  if question in M:
    i = M.index(question)
    M.pop(i)
  M.append(question)
  if quest_memory>0 : M=M[-quest_memory:]
  Q=reversed(M)
  question = ''.join(Q)
  dialog_step(prolog, question, gm, qgm, fNameNoSuf,log)


# step in a dialog agent based given file and question
def dialog_step(prolog,question,gm,qgm,fNameNoSuf,log) :
    query_to_prolog(question,gm,qgm,fNameNoSuf)
    rs=prolog.query("ask('" + fNameNoSuf + "'"  + ",Key)")
    answers=[pair['Key'] for pair in rs]
    log.append((question,answers))
    if not answers: say("Sorry, I have no good answer to that.")
    else :
      for answer in answers :
        say(answer)
    print('')
      
   
def sink(generator) :
  for _ in generator : pass


def getNERs(ws):
  parser = CoreNLPParser(url=parserURL, tagtype='ner')
  ts = parser.tag(ws)
  for t in ts :
    if t[1] !='O' :
      yield t


def ner_test():
  print(list(
    getNERs(['today','MIT', 'Stanford','London','Austin, Texas', 'Permian Basin'])))
    
# sends dependency triples to Prolog, as rececived from Parser  
def triples_to_prolog(pref,qgm,f) :
    ctr=0
    for g in qgm.gs :
      for x in g.triples() :
        (fr,ft),r,(to,tt)=x
        print(pref+'dep',end='',file=f)
        print((ctr,fr,ft,r,to,tt),end='',file=f)
        print('.',file=f)
      ctr+=1

def sents_to_prolog(pref, qgm, f):
  s_ws_gen=dr.sent_words(qgm)
  for s_ws in s_ws_gen:
    print(pref + 'sent', end='', file=f)
    print(s_ws, end='', file=f)
    print('.', file=f)


def ners_to_prolog(pref, qgm, f):
  s_ws_gen=dr.sent_words(qgm)
  for s_ws in s_ws_gen:
    s,ws=s_ws
    ners=list(enumerate(getNERs(ws)))
    if ners:
      print(pref + 'ner', end='', file=f)
      print((s,ners), end='', file=f)
      print('.', file=f)


# sends summaries to Prolog    
def sums_to_prolog(pref,k,qgm,f) :
    if pref : return
    for sent in qgm.bestSentences(k) :
      print('summary',end='',file=f)
      print(sent,end='',file=f)
      print('.',file=f)   
 
# sends keyphrases to Prolog    
def keys_to_prolog(pref,k,qgm,f) :
    if pref : return
    for kw in qgm.bestWords(k) :
      print(pref+"keyword('",end='',file=f)
      print(kw,end="')",file=f)
      print('.',file=f)   
      
# sends edges of the graph to Prolog
def edges_to_prolog(pref,qgm,f) :
    for ek in qgm.edgesInSent() :
      e,k=ek
      a,aa,r,b,bb=e
      e = (k,a,aa,r,b,bb)
      print(pref+'edge',end='',file=f)
      print(e,end='',file=f)
      print('.',file=f)
    if pics=='yes' and pref :
      dr.query_edges_to_dot(qgm)

# generic Prolog predicate maker
def facts_to_prolog(pref,name,facts,f) :
  if isinstance(facts,dict) :
    facts=facts.items()
  for fact in facts :
    print(pref+name,end='',file=f)
    print(fact,end='',file=f)
    print('.',file=f)
  print('',file=f)

    
# sends the computed ranks to Prolog
def ranks_to_prolog(pref,qgm,f) :
    ranks=qgm.pagerank()
    facts_to_prolog(pref, 'rank', ranks, f)


# sends the words to lemmas table to Prolog      
def w2l_to_prolog(pref,qgm,f) :
    tuples=qgm.words2lemmas
    for r in tuples :
      print(pref+'w2l',end='',file=f)
      print(r,end='',file=f)
      print('.',file=f)   

# sends svo realtions to Prolog
def svo_to_prolog(pref,qgm,f) :
    rs=qgm.bestSVOs(100)
    facts_to_prolog(pref, 'svo', rs, f)
      
# sends a similarity relation map to Prolog    
def sims_to_prolog(pref,gm,qgm,f) :
  #print(qgm.words)
  for qs in qgm.words2lemmas :
    qw,ql,qt=qs
    for cs in gm.words2lemmas :
      cw,cl,ct=cs
      if ql!=cl : # and qt==ct:
        if is_similar(ql,qt,cl,ct):
          print('query_sim',end='',file=f)
          print((ql,qt,cl,ct),end='',file=f)
          print('.',file=f)

# sends a similarity relation map to Prolog    
def rels_to_prolog(pref,gm,qgm,f) :
  def sentId(touple) :
    return touple[3]
  pr=gm.pagerank()
  ws=dict()
  rels=set()
  i=0
  for w in pr :
    if dr.isWord(w) :
      ws[w]=i
      i+=1
  for qs in qgm.words2lemmas :
    _,ql,qt=qs
    wn_tag=pos2tag(qt)
    if wn_tag!='n' : continue
    hypers=wn_hyper(3,ql,wn_tag)
    hypos=wn_hypo(3,ql,wn_tag)
    meros=wn_mero(3,ql,wn_tag)
    holos=wn_holo(3,ql,wn_tag)
    for h in hypers :
      if h in ws :
        rels.add((ql,'is_a',h,-ws[h])) # order in ranks = -ws[h]
    for h in hypos :
      if h in ws :
        rels.add((h,'is_a',ql,-ws[h]))
    for h in meros :
      if h in ws :
        rels.add((h,'part_of',ql,-ws[h]))
    for h in holos :
      if h in ws :
        rels.add((ql,'part_of',h,-ws[h]))  
  rels=sorted(rels,key=sentId,reverse=True)
  facts_to_prolog(pref,'rel',rels,f)

# exporting to Prolog files needed to answer query

def export_to_prolog(fNameNoSuf,OutF=None) :
  gm=dr.GraphMaker()
  gm.load(fNameNoSuf+'.txt')
  if not OutF :
    OutF = fNameNoSuf
  to_prolog('',gm,gm,OutF)
  return gm

def params_to_prolog(pref,f) :
  rels=[('quest_memory',quest_memory),
        ('max_answers', max_answers),
        ('repeat_answers',repeat_answers),
        ('personalize',personalize),
        ('by_rank', by_rank)
        ]
  facts_to_prolog(pref, 'param', rels, f)

def personalize_for_query(gm, qgm, sk, wk):
    query_dict = dr.pers_dict(qgm)
    ranks = gm.rerank(query_dict)

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
def query_to_prolog(text,gm,qgm,fNameNoSuf) :
  qgm.digest(text)
  qfName=fNameNoSuf+'_query'
  to_prolog('query_',gm,qgm,qfName)

def personalized_to_prolog(pref,gm,qgm,personalize,f) :
  count=personalize
  #print("COUNT",count)
  assert(isinstance(count,int))
  (sents,words) = personalize_for_query(gm,qgm,count,count)
  facts_to_prolog(pref, 'pers_sents', sents, f)
  facts_to_prolog(pref, 'pers_words', words, f)

# sends several fact predicates to Prolog
# small files are used that the pyswip activated Prolog will answer
# the pref='query_' marks file names with query_, while
# the empty prefix pref='' marks realtions describing a document
def to_prolog(pref,gm,qgm,fNameNoSuf) :
  with open(fNameNoSuf+'.pro','w') as f :
    triples_to_prolog(pref,qgm,f)
    print(' ',file=f)
    edges_to_prolog(pref,qgm,f)    
    print(' ',file=f)
    ranks_to_prolog(pref,qgm,f)  
    print(' ',file=f)
    w2l_to_prolog(pref,qgm,f)   
    print(' ',file=f)
    sents_to_prolog(pref,qgm,f) 
    print(' ',file=f)
    ners_to_prolog(pref, qgm, f)
    print(' ', file=f)
    svo_to_prolog(pref,qgm,f)   
    print(' ',file=f) 
    if pref : # query only
        #sims_to_prolog(pref,gm,qgm,f)
        rels_to_prolog(pref,gm,qgm,f) # should be after svo!
        if personalize>0 :
           personalized_to_prolog(pref,gm,qgm,personalize,f)
        params_to_prolog(pref, f)

    else : # document only
        sums_to_prolog(pref,10,qgm,f)
        print(' ',file=f)  
        keys_to_prolog(pref,10,qgm,f)
        print(' ',file=f)


def ptest():
  f = 'examples/bfr'
  qf = f + '_query.pro'
  gm = export_to_prolog(f)
  prolog = Prolog()
  prolog.consult(f + '.pro')
  q = prolog.query('listing(dep)')
  next(q)
  q.close()
  qgm = dr.GraphMaker()
  query_to_prolog('What is the BFR?', gm, qgm, f)
  prolog.consult(qf)
  q = prolog.query('listing(query_sent)')
  next(q)
  q.close()


def all_ts():
  for i in range(0, 10):
    f = 't' + str(i)
    eval(f + "()")


def chat(FNameNoSuf):
  return dialog_about('examples/' + FNameNoSuf, None)


def pdf_chat(FNameNoSuf):
  return  pdf_chat_with("pdfs", FNameNoSuf)


def pdf_chat_with(Folder, FNameNoSuf, about=None):
  fname = Folder + "/" + FNameNoSuf
  dr.pdf2txt(fname + ".pdf")
  return  dialog_about(fname, about)


def pdf_quest(Folder, FNameNoSuf, QuestFileNoSuf):
  Q = []
  qfname = Folder + "/" + QuestFileNoSuf + ".txt"
  qs = list(ev.file2seq(qfname))
  return  pdf_chat_with(Folder, FNameNoSuf, about=qs)


def txt_quest(Folder, FNameNoSuf, QuestFileNoSuf):
  Q = []
  qfname = Folder + "/" + QuestFileNoSuf + ".txt"
  qs = list(ev.file2seq(qfname))
  # print('qs',qs)
  return  dialog_about(Folder + "/" + FNameNoSuf, qs)


def q0():
  d=txt_quest('examples', 'tesla', 'tesla_quest')
  print('LOG',d)


def q1():
  d=txt_quest('examples', 'bfr', 'bfr_quest')
  print('LOG',d)



def t0():
  dialog_about('examples/tesla',
               "How I have a flat tire repaired?")


def t0a():
  dialog_about('examples/tesla',
      "How I have a flat tire repaired?  \
      Do I have Autopilot enabled? \
      How I navigate to work? \
      Should I check tire pressures?")


def t1():
  d=dialog_about('examples/bfr',
               "What space vehicles SpaceX develops?")
  print('Sentece IDs: ',d)


def t2():
  # dialog_about('examples/bfr')
  dialog_about('examples/hindenburg',
               "How did the  fire start on the Hindenburg?")


def t3():
  dialog_about('examples/const',
  # "How many votes are needed for the impeachment of a President?"
        'How can a President be removed from office?'
  )


def t4():
  dialog_about('examples/summary',
               "How we obtain summaries and keywords from dependency graphs?")


def t5():
  dialog_about('examples/heaven',
               "What does the Pope think about heaven?")


def t6():
  dialog_about('examples/einstein',
               "What does quantum theory tell us about our \
                description of reality for an observer?")


def t7():
  dialog_about('examples/kafka',
               # "What does the doorkeeper say about entering?"
               "Why does K. want access to the law at any price?"
               )


def t8():
  dialog_about('examples/test',
               "Does Mary have a book?")


def t9():
  dialog_about('examples/relativity',
               "What happens to light in the presence of gravitational fields?")


def t10():
  pdf_chat_with('pdfs', 'textrank',
                about='What are the applications of TextRank? \
      How sentence extraction works? What is the role of PageRank?')


def t11():
  with open('examples/texas_quest.txt', 'r') as f:
    qs = list(l[:-1] for l in f)
    return dialog_about('examples/' + "texas",qs)


def p1():
  fd = ev.doc_dir
  fn = "1039329"
  fname = fd + fn
  export_to_prolog(fname, fd + "pro/" + fn)


ppp=print

#all_ts()

'''
todo:

remove: svo(x,rel,x)


'''

if __name__ == '__main__'  :
  pass
