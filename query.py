from params import *
import deepRank as dr
import eval as ev
from collections import defaultdict

def multi_dict():
  return defaultdict(set)

class DialogAgent(dr.GraphMaker):
  def __init__(self):
    super().__init__()
    self.source = None
    self.target = None

  def source_target_pair(self):
    for e, k in self.edgesInSent():
      f, ft, r, t, tt = e
      yield f, (k,t, (ft, r, tt))

  def target_source_pair(self):
    for e, k in self.edgesInSent():
      f, ft, r, t, tt = e
      yield t, (k, f, (ft,r,tt))

  def source_index(self):
    m = multi_dict()
    for x, y in self.source_target_pair():
      m[x].add(y)
    return m

  def target_index(self):
    m = multi_dict()
    for x, y in self.target_source_pair():
      m[x].add(y)
    return m

  def index(self):
    if not self.source:
      self.source = self.source_index()
    if not self.target:
      self.target = self.target_index()

  def stats(self):
    return (len(self.source), len(self.target))

  def chat_about(self,fname,question=None):
    sk=6
    ak=3
    self.load(fname)
    pr=self.pagerank()
    #chat(self.index()
    #for x in self.source.items() : print(x,'\n')
    QA=DialogAgent()

    summary=list(self.bestSentences(sk))
    sents={s for s,_ in summary}
    print('SUMMARY',sents)
    dr.print_summary(summary)
    self.index()
    #print('SELF',self.stats())
    db=set(self.source).union(set(self.target))
    #print('DB',len(db))

    def query(QA, text, echo=False):
      if echo: print('?--', text)
      QA.digest(text)
      pr = QA.pagerank()
      qpr = self.rerank(pers=pr)

      QA.index()
      #print('QA', QA.stats())
      qa=set(QA.source).union(set(QA.target))
      shared = { x for x in qa.intersection(db)
                   if dr.maybeWord(x) and not dr.isStopWord(x)
               }
      #print('SHARED',shared)

      good=set()
      for w in shared :
        for (s,k,_) in self.source[w] :
          if dr.isSent(k) : good.add(k)
        for (s, k, _) in self.target[w]:
          if dr.isSent(k) : good.add(k)

      def inhabits(x) : return x in good
      answers = list(self.bestSentencesByRank(ak,filter=inhabits))
      new_answers=[(s,a) for s,a in answers if s not in sents]
      if new_answers : answers=new_answers

      dr.print_summary(answers)
      if pics == 'yes':
        dr.query_edges_to_dot(QA)
        self.toDot(12, dr.maybeWord, svo=True, show=True)

    if isinstance(question, list):
      for q in question:
        query(QA,q,echo=True)
    elif question :
      query(QA,question,echo=True)
    else :
      while (True):
        question = input('?-- ')
        if not question: break
        query(QA,question)


def chat(fname) :
  DialogAgent().chat_about("examples/"+fname+".txt")

def dialog_about(fname,query=None) :
  DialogAgent().chat_about(fname+".txt",question=query)

def ranked_txt_quest(Folder, FNameNoSuf, QuestFileNoSuf):
  Q = []
  qfname = Folder + "/" + QuestFileNoSuf + ".txt"
  qs = list(ev.file2seq(qfname))
  print('qs',qs)
  dialog_about(Folder+"/"+FNameNoSuf,qs)

def t0():
  dialog_about('examples/tesla',
               "How I have a flat tire repaired?")


def t0a():
  dialog_about('examples/tesla',
               "How I have a flat tire repaired?  \
               Do I have Autopilot enabled? \
               How I navigate to work? Should I check tire pressures?")

def t0b() :
  ranked_txt_quest('examples','tesla','quests')

def t1():
  dialog_about('examples/bfr',
               "What space vehicles SpaceX develops?")

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


def go() :
  t1()
  t2()
  t3()
  t4()
  t5()
  t6()
  t7()
  t8()
  t9()
  t0()

if __name__ == '__main__'  :
  pass
  #t4()


