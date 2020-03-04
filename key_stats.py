from nltk.metrics.scores import recall,precision,f_measure
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import nltk

ps=PorterStemmer()
lm=WordNetLemmatizer() 

def kset_stat(silvs,golds) :
  s1 = set(map(to_root,golds))
  s2 = set(map(to_root,silvs))
  #print(s1,s2)
  p=precision(s1,s2)
  r=recall(s1,s2)
  f=f_measure(s1,s2)
  if not (p and r and f) : return {'p':0,'r':0,'f':0}
  return {'p':p,'r':r,'f':f}

def kstat(silver,gold) :
  
  silvs=nltk.word_tokenize(silver)
  golds=nltk.word_tokenize(gold)
  
  return kset_stat(silvs,golds)
  
def to_root(w) :
  return ps.stem(lm.lemmatize(w))
  
def go1() :
  gold=('cat','dog','mouse','owls')  
  silver=('cats','mice','rats') 
  print(kset_stat(silver,gold))

def go2() :
  golds={'polynomi', 'irreduc', 'continu', 'system', 'gener', 'geometri', 'of', 'embed', 'algebra', 'homotopi', 'solut', 'point', 'compon', 'numer'}
  silvs={'assum', 'algorithm', 'system', 'path', 'wit', 'solut', 'point', 'set', 'compon'}
  print(kset_stat(silvs,golds))
  
def go() :
  gold='cat dog mouse owls' 
  silver='cats mice rats' 
  print('key_test',kstat(silver,gold))
  
# (0.6666666666666666, 0.5, 0.5714285714285714)

if __name__ == '__main__' :
  #go()
  pass