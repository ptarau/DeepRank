import rouge

def rstat(all_hypothesis,all_references) :
  for aggregator in ['Individual'] : #['Avg', 'Best', 'Individual']:
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=2,
                           limit_length=False,
                           #length_limit=300,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
    scores = evaluator.get_scores(all_hypothesis, all_references)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
      yield results
      # come out for rouge n=1,n=2,l,w

def hyps_and_refs() :
    hypothesis_1 = "The dog hates the cat. Th cat hates the mouse."
    references_1 = "The dog gave a book to the cat. The cat gave a book to the mouse. The mouse ate the book. They hate each other since then."
    return (hypothesis_1,references_1)

def go() :
  hs,ds=hyps_and_refs()
  for r in rstat(hs,ds) : print(r)

if __name__=='__main__' :
  go()
    