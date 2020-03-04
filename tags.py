'''Alphabetical listing

UDP POS tags:

ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary verb
CONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other

'''

postags={
      'CC':'Coordinating conjunction',
      'CD':'Cardinal number',
      'DT': 'Determiner',
      'EX ': 'Existential there',
      'FW': 'Foreign word',
      'IN': 'Preposition or subordinating conjunction',
      'JJ': 'Adjective',
      'JJR': 'Adjective, comparative',
      'JJS': 'Adjective, superlative',
      'LS': 'List item marker',
      'MD': 'Modal',
      'NN': 'Noun, singular or mass',
      'NNS': 'Noun, plural',
      'NNP': 'Proper noun, singular',
      'NNPS': 'Proper noun, plural',
      'PDT': 'Predeterminer',
      'POS': 'Possessive ending',
      'PRP': 'Personal pronoun',
      'PRP$': 'Possessive pronoun',
      'RB': 'Adverb',
      'RBR': 'Adverb, comparative',
      'RBS': 'Adverb, superlative',
      'RP': 'Particle',
      'SYM': 'Symbol',
      'TO': 'to',
      'UH': 'Interjection',
      'VB': 'Verb, base form',
      'VBD': 'Verb, past tense',
      'VBG': 'Verb, gerund or present participle',
      'VBN': 'Verb, past participle',
      'VBP': 'Verb, non-3rd person singular present',
      'VBZ': 'Verb, 3rd person singular present',
      'WDT': 'Wh-determiner',
      'WP': 'Wh-pronoun',
      'WP$': 'Possessive wh-pronoun',
      'WRB': 'Wh-adverb'
    }   

rels= {
      'acl': 'clausal modifier of noun (adjectival clause)',
      'advcl': 'adverbial clause modifier',
      'advmod': 'adverbial modifier',
      'amod': 'adjectival modifier',
      'appos': 'appositional modifier',
      'aux': 'auxiliary',
      'case': 'case marking',
      'cc': 'coordinating conjunction',
      'ccomp': 'clausal complement',
      'clf': 'classifier',
      'compound': 'compound',
      'conj': 'conjunct',
      'cop': 'copula',
      'csubj': 'clausal subject',
      'dep': 'unspecified dependency',
      'det': 'determiner',
      'discourse': 'discourse element',
      'dislocated': 'dislocated elements',
      'expl': 'expletive',
      'fixed': 'fixed multiword expression',
      'flat': 'flat multiword expression',
      'goeswith': 'goes with',
      'iobj': 'indirect object',
      'list': 'list',
      'mark': 'marker',
      'nmod': 'nominal modifier',
      'nsubj': 'nominal subject',
      'nummod': 'numeric modifier',
      'obj': 'object',
      'obl': 'oblique nominal',
      'orphan': 'orphan',
      'parataxis': 'parataxis',
      'punct': 'punctuation',
      'reparandum': 'overridden disfluency',
      'root': 'root',
      'vocative': 'vocative',
      'xcomp': 'open clausal complement'
      }
      