import glob
import sys
import os
import deepRank as dr
import rouge_stats as rs
import key_stats as ks
from itertools import islice

from params import *

if prod_mode :
  data_dir='dataset/Krapivin2009/'
else :
  data_dir='dataset/small/'
  
doc_dir=data_dir+'docsutf8/'
keys_dir=data_dir+'keys/'
abs_dir=data_dir+'abs/'
all_doc_files = sorted(glob.glob(doc_dir+"*.txt"))

if max_docs :
  doc_files=list(islice(all_doc_files,max_docs))
else :
  doc_files=all_doc_files

if prod_mode :
  out_abs_dir  = "out/abs/"
  out_keys_dir = "out/keys/"
else :
  out_abs_dir  = "test/abs/"
  out_keys_dir = "test/keys/"

# clean output directories
def clean_all() :
  clean_path(out_abs_dir)
  clean_path(out_keys_dir)

# clean files at given directory path 
def clean_path(path) :
  os.makedirs(path,exist_ok=True)

  files = glob.glob(path+"/*")
  for f in files:
    os.remove(f)
   
# extract triple (title,abstract,body) with refs trimmed out
def disect_doc(doc_file) :
  title=[]
  abstract=[]
  body=[]
  mode=None
  with open(doc_file) as f:
    for line in f:
      if line.startswith('--T')   : mode='TITLE'
      elif line.startswith('--A') : mode ='ABS'
      elif line.startswith('--B') : mode = 'BODY'
      elif line.startswith('--R'): mode = 'DONE'
      else :
        if   mode=='TITLE': title.append(line.strip()+' ')
        elif mode=='ABS'  : abstract.append(line.strip()+' ')
        elif mode=='BODY' : body.append(line.strip()+' ')
        elif mode=='DONE' : break
  return {'TITLE':title,'ABSTRACT':abstract,'BODY':body}

# process string text give word count,sentence count and filter
def runWithText(text,wk,sk,filter) :
  gm=dr.GraphMaker()
  gm.digest(text)
  keys= gm.bestWords(wk)
  sents=[s for (_,s) in gm.bestSentences(sk)]
  #keys_text=interleave_with('\n','\n',keys)
  #sents_text=interleave_with('\n','\n',sents)
  #return (keys_text,sents_text)
  nk=gm.nxgraph.number_of_nodes()
  vk=gm.nxgraph.number_of_edges()
  return (keys,sents,nk,vk)
  
#  extract the gold standard abstracts from dataset  
def fill_out_abs() :
   for doc_file in doc_files :
     d=disect_doc(doc_file)
     abstract=d['ABSTRACT']
     text=''.join(abstract)
     abs_file=abs_dir+dr.path2fname(doc_file)
     print('abstract extraced to: ',abs_file)
     string2file(abs_file,text)

     
# turns a sequence/generator into a file, one line per item yield     
def seq2file(fname,seq) :
  xs=map(str,seq)
  ys=interleave_with('\n','\n',xs)
  text=''.join(ys)
  string2file(fname,text)

# turns a file into a (string) generator yielding each of its lines
def file2seq(fname) :
   with open(fname,'r') as f :
     for l in f : yield l.strip()

# turns a string into given file
def string2file(fname,text) :
  with open(fname,'w') as f :  
    f.write(text)

# turns content of file into a string
def file2string(fname) :
  with open(fname,'r') as f :
    s = f.read()
    return s.replace('-',' ')

# interleaves list with separator
def interleave(sep,xs) :
  return interleave_with(sep,None,xs)
  
def interleave_with(sep,end,xs) :
  def gen() :
    first=True
    for x in xs : 
      if not first : yield sep
      yield x
      first=False
    if end : yield(end)
      
  return ''.join(gen())

# extracts keys and abstacts from resource directory  
def extract_keys_and_abs(full,wk,sk) :
  clean_all()
  for path_file in doc_files :
    doc_file=dr.path2fname(path_file)
    try :
      d=disect_doc(path_file)
      title = d['TITLE']
      abstract=d['ABSTRACT']
      body = d['BODY']
      text_no_abs=''.join(title + [' '] + body)
  
      if full : text=''.join(title + [' '] + abstract + [' '] + body)
      else : text=''.join(title + [' '] + body)

      (keys,xss,nk,ek) = runWithText(text,wk,sk,dr.isWord)
      print(doc_file,'nodes:',nk,'edges:',ek)  # ,title)
      exabs = map(lambda x: interleave(' ',x),xss)
      kf=out_keys_dir+doc_file
      af=out_abs_dir+doc_file
      seq2file(kf,keys)
      seq2file(af,exabs)
    except :
      print('*** FAILING on:',doc_file,'ERROR:',sys.exc_info()[0])


# apply Python base rouge to abstracts from given directory
def eval_with_rouge(i) :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=abs_dir+fname
    abs_name=out_abs_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(ref_name)   
    silver=file2string(abs_name)
    k=0
    for res in rs.rstat(silver,gold) :
      if k==i:    
        d=res[0]
      
        px=d['p'][0]
        rx=d['r'][0]
        fx=d['f'][0]
    
        p.append(px)
        r.append(rx)
        f.append(fx)
        
      elif k>i : break
      k+=1
    if trace_mode : print('  ABS ROUGE MOV. AVG',i,fname,avg(p),avg(r),avg(f))
  rouge_name=(1,2,'l','w')  
  print ("ABS ROUGE",rouge_name[i],':',avg(p),avg(r),avg(f))

# our own 
def eval_abs() :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=abs_dir+fname
    abs_name=out_abs_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(ref_name)
    silver=file2string(abs_name)
    #print(gold)
    #print(silver)
    d=ks.kstat(silver,gold)
    if not d :
      print('FAILING on',fname)
      continue
    if trace_mode: print('  ABS SCORE:',d)
    px=d['p']
    rx=d['r']
    fx=d['f']
    if px and rx and fx :
      p.append(px)
      r.append(rx)
      f.append(fx)
    if trace_mode : print('  ABS MOV. AVG',fname,avg(p),avg(r),avg(f))
  print ("ABS SCORES  :",avg(p),avg(r),avg(f))

  
# 0.22434732994628803 0.24271988542882067 0.22280040709372084
def eval_keys() :
  f=[]
  p=[]
  r=[]  
  for doc_file in doc_files : 
    fname=dr.path2fname(doc_file)
    ref_name=keys_dir+fname
    keys_name=out_keys_dir+fname
    #if trace_mode : print(fname)
    gold=file2string(txt2key(ref_name))   
    silver=file2string(keys_name)
    #print(gold)
    #print(silver)
    d=ks.kstat(silver,gold)
    if not d :
      print('FAILING on',fname)
      print('SILVER',silver)
      print('GOLD',gold)
      continue
    if trace_mode : print('  KEYS',d)
    px=d['p']
    rx=d['r']
    fx=d['f']
    p.append(px)
    r.append(rx)
    f.append(fx)
    #if trace_mode : print('  KEYS . AVG:',fname,avg(p),avg(r),avg(f))
  print('KEYS SCORES :',avg(p),avg(r),avg(f))
  
  
def txt2key(fname) :
  return fname.replace('.txt','.key')
    
def avg(xs) :
  s=sum(xs)
  l=len(xs)
  if 0==l : return None
  return s/l  

######### main evaluator #############
def go() :

  #fill_out_abs


  def showParams() :
    print('wk',wk,'sk',sk,'\n'
          'with_full_text = ',with_full_text,'\n',
          'prod_mode = ' ,prod_mode,'\n',
          'max_docs = ',max_docs,'\n',
          'noun_defs = ',noun_defs,'\n',
          'all_recs =',all_recs,'\n'
          )

  print("STARTING")
  showParams()
  extract_keys_and_abs(with_full_text, wk, sk)
  eval_keys()
  eval_abs()
  eval_with_rouge(0)  # 1
  eval_with_rouge(1)  # 2
  eval_with_rouge(2)  # l
  eval_with_rouge(3)  # w
  print('DONE')
  showParams()

if __name__ == '__main__' :
  pass
  #go()

'''
sqrt
KEYS SCORES : 0.27416666666666667 0.34694444444444444 0.29275340952551887
ABS SCORES  : 0.36118382771545704 0.5042334736635918 0.41332132103435465
ABS ROUGE 1 : 0.3933145118929155 0.5184928165336313 0.43482299877528385
ABS ROUGE 2 : 0.16357039735115525 0.232778796923158 0.1858206751162993
ABS ROUGE l : 0.34487947667156577 0.442321878246675 0.3792731992052987
ABS ROUGE w : 0.1964525340859074 0.09936084572373194 0.12626315636206803

log
KEYS SCORES : 0.27416666666666667 0.34694444444444444 0.29275340952551887
ABS SCORES  : 0.37839101818507703 0.5385774853321073 0.4364521548054726
ABS ROUGE 1 : 0.4046478727360066 0.5505292131459584 0.45296490614873336
ABS ROUGE 2 : 0.19829718998915621 0.2972863472526791 0.2295444270832006
ABS ROUGE l : 0.36403723150031414 0.4788527681008241 0.40464878815939126
ABS ROUGE w : 0.21642668012136568 0.11444788050837104 0.1430266946676839

log all_recs=False
KEYS SCORES : 0.26527777777777783 0.34555555555555556 0.28860263403607056
ABS SCORES  : 0.39251626829955083 0.5245116880849674 0.439518967424411
ABS ROUGE 1 : 0.42287435538537965 0.5259024228548685 0.4524099817168163
ABS ROUGE 2 : 0.21639132498349145 0.2935673723686802 0.23872381307767448
ABS ROUGE l : 0.38221906158068136 0.46994688459791806 0.41067624936976016
ABS ROUGE w : 0.23062363261393387 0.11227662395080236 0.14346144225940838

'''
