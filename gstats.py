# graph theoretical stats of text graph
import networkx as nx

from networkx.algorithms.smallworld import sigma,omega

import networkx.algorithms.distance_measures as dm
import networkx.algorithms.components as co
import networkx.algorithms.cluster as cl

import deepRank as dr

def runWith(fname) :
  print(fname)
  gm=dr.GraphMaker()
  gm.load(fname)
  # dpr=gm.pagerank()
  dg=gm.graph()
  #for x in dg: print('VERT::', x)

  print('nodes:', dg.number_of_nodes())
  print('edges:', dg.number_of_edges())

  comps=nx.strongly_connected_components(dg)

  print('strongly connected components:',len(list(comps)))

  c = max(nx.strongly_connected_components(dg), key=len)
  mg=dg.subgraph(c)

  print('attracting components:', co.number_attracting_components(dg))
  print('number_weakly_connected_components:',co.number_weakly_connected_components(dg))

  print('Transitivity:',cl.transitivity(dg))

  return

  e=dm.eccentricity(mg)

  dprint('ecc:', e)

  cent=dm.center(mg,e=e)
  print('CENTER',cent)

  p=dm.periphery(mg,e=e)

  print('perif:', len(list(e)))

  #dprint('perif:', e)

  print('diameter:', dm.diameter(nx.Graph(mg)))
  print('radius:', dm.radius(nx.Graph(mg)))

  g = nx.Graph(dg)
  print('omega:', omega(g))
  print('sigma:', sigma(g))

def dprint(mes,d) :
  print('--------',mes,'-----------')
  for x in d :
    print(x,':',d[x])
  print('')

def go() :
  runWith('examples/const.txt')

go()

