import os
from deep_talk.min_qpro import *
from deep_talk.query import qgo

def ptest():
  f = 'examples/bfr'
  qf = f + '_query.pro'
  gm = export_to_prolog(f)



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
  print('Sentence IDs: ',d)


def t2():
  # dialog_about('examples/bfr')
  dialog_about('examples/hindenburg',
               "When did the  fire start on the Hindenburg?")


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


def t10a():
  pdf_chat_with('pdfs', 'textrank',
                about='What are the applications of TextRank? \
      How sentence extraction works? What is the role of PageRank?')

def t10():
  d=txt_quest('examples', 'textrank', 'textrank_quest')
  print('LOG',d)

def t11():
  d=txt_quest('examples', 'texas', 'texas_quest')
  print('LOG',d)

def t12():
  d=txt_quest('examples', 'heli', 'heli_quest')
  print('LOG',d)

def t13():
  d=txt_quest('examples', 'red', 'red_quest')
  print('LOG',d)

def t14():
  d=txt_quest('examples', 'covid', 'covid_quest')
  print('LOG',d)


def all_ts():
  for i in range(0, 15):
    f = 't' + str(i)
    eval(f + "()")

def qtests() :
  from deep_talk.query import t1
  t13()

if __name__=='__main__' :
  #t1()
  #all_ts()
  qgo()
  pass

