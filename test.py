import os
from textcrafts.deepRank import maybeWord, isAny, pdf2txt
from textcrafts import GraphMaker, craft_params

params= craft_params()

def testx() :
  gm = GraphMaker(text='The cat sits on the mat.')
  print(gm.triples())
  print(gm.lemmas())
  print(gm.words())
  print(gm.tags())


# interactive read, parse, show labeled edges loop
def testy():
    gm= GraphMaker(text='The cat walks. The dog barks.')
    for g in gm.gs:
      print(g)
    for f, ft, r, t, tt in gm.edges():
      print(f, '->', r, '->', t)
    pr = gm.pagerank()
    for w in pr.items(): print(w)


def runWithFilter(fileName,filter=maybeWord):
    gm = GraphMaker(file_name=fileName)
    fname=os.path.splitext(fileName)[0]
    dotName = fname+".gv"
    cloudName=fname+"_cloud.pdf"
    gm.toDot(params.dot_count, filter, svo=True, fname=dotName, show=params.show)
    gm.kshow(params.dot_count,file_name=cloudName,show=params.show)
    return gm

def test0():  # might take 1-2 minutes
    gm = runWithFilter('examples/tesla.txt')
    return gm


def test1():
    gm = runWithFilter('examples/bfr.txt')
    return gm


def test2():
    wk, sk = 3, 3
    gm = runWithFilter('examples/hindenburg.txt')
    return gm


def test3():
    gm = runWithFilter('examples/const.txt')
    return gm


def test4():
    gm = gm = runWithFilter('examples/summary.txt',  filter=maybeWord)
    return gm


def test5():
    gm = runWithFilter('examples/heaven.txt')
    return gm


def test6():
    gm = runWithFilter('examples/einstein.txt')
    return gm


def test7():
    gm = runWithFilter('examples/kafka.txt')
    return gm


def test8():
    gm = runWithFilter('examples/test.txt')
    return gm


def test9():
    gm = runWithFilter('examples/relativity.txt')
    return gm


def test10():
    gm = runWithFilter('examples/cats.txt')
    return gm


def test11():
    gm = runWithFilter('examples/wasteland.txt')
    return gm


def test12():
    fname = "../pdfs/textrank"
    pdf2txt(fname+".pdf")
    gm = runWithFilter(fname+".txt")
    return gm


def test13():
  gm = runWithFilter('examples/red.txt')
  return gm


def testx():
    gm = GraphMaker(text='The cat sits. The dog barks.')
    for e in gm.triples():
            print(e)
    for f, ft, r, t, tt in gm.edges():
        print(f, '->', r, '->', t)
    pr = gm.pagerank()
    for w in pr.items():
        print(w)
    return gm

def go() :
  print(testx())
  print(testy())
  print(test1())
  print(test2())
  print(test3())
  print(test4())
  print(test5())
  print(test6())
  print(test7())
  print(test8())
  print(test9())
  print(test10())
  print(test11())
  #print(test12())
  print(test0())


if __name__=='__main__' :
  #print('TESTING')
  print(test13())
