from graphviz import Digraph as DotGraph
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def showGraph(dot, show=True, file_name='textgraph.gv'):
  dot.render(file_name, view=show)

'''
def gshow0(g, file_name='textgraph.gv', show=True):
  dot = DotGraph()
  for e in g.edges():
    f, t = e
    # w = g[f][t]['weight']
    w = ''
    dot.edge(str(f), str(t), label=str(w))
  dot.render(file_name, view=show)
'''

def gskim(g,attr=None,roots=None,filter=lambda x : isinstance(x,str)) :
  newg=nx.DiGraph()
  if roots==None: roots=[x for x in g.nodes() if filter(x)]
  for x in roots:
    for y in g[x] :
      if attr:
        val=g[x][y][attr]
        newg.add_edge(x,y,attr=val)
      else :
        newg.add_edge(x, y)

def gshow(g, attr=None, file_name='temp.gv', show=1):

  size=g.number_of_edges()
  nsize=g.number_of_nodes()

  if size < 3 :
    print('GRAPH TOO SMALL TO SHOW:', file_name, 'nodes:',nsize,'edges:', size)
    return
  elif size <300 :
    print('SHOWING:',file_name, 'nodes:',nsize,'edges:', size)
  else:
    print('TOO BIG TO SHOW:',file_name, 'nodes:',nsize,'edges:', size)
    return
  dot = DotGraph()
  for e in g.edges():
    f, t = e
    if not attr : w= ''
    else :
      w = g[f][t].get(attr)
      if not w : w=''
    dot.edge(str(f), str(t), label=str(w))
  dot.render(file_name, view=show>1)

def show_ranks(rank_dict,file_name="cloud.pdf",show=True) :
  if not show : return
  cloud=WordCloud(width=800,height=400)
  cloud.fit_words(rank_dict)
  f=plt.figure()
  plt.imshow(cloud, interpolation='bilinear')
  plt.axis("off")
  # plt.show()
  f.savefig(file_name,bbox_inches='tight')
  plt.close('all')

if __name__=="__main__":
  d = {'a': 0.1, 'b': 0.2, 'c': 0.33, 'd': 0.2}
  show_ranks(d)
