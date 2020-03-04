###### parameters #######
import math

# Graph building, parsing ranking

parserURL='http://localhost:9000'
abstractive='no'

## for EVALUATION

# sets max s number of documents to be processed, all if None
max_docs = None
# resource directories, for production and testing at small scale
prod_mode=False
# shows moving averages if on
trace_mode=False
# if true abstracts are not trimmed out from documents
with_full_text = True

# for LINKS, RANKING, SUMMARIES AND KEYPHRASES

# sets link addition parameters
all_recs  = True  # sentence recommendatations
giant_comp = False # only extract from giant comp
noun_defs = True

noun_self = False

# number of keyphrases
wk = 5
# number of summary sentences
sk = 3


## for Dialog Engines qpro.py and query.py

quiet=True
summarize=True
quest_memory=1
max_answers=3
repeat_answers='yes'
by_rank='yes'
personalize=50
pics='no'

# formula for adjusting rank of long or short sentences
def adjust_rank(rank,length,avg) :
   #adjust = 1 + math.sqrt(1 + abs(length - avg))
   adjust = 1 + math.log(1+abs(length-avg))
   newr=rank/adjust
   #print('ADJUST',adjust,length,avg)
   return newr
