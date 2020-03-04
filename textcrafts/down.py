import ssl
import sys
import os
import nltk

def ensure_nlk_downloads() :
  sout = sys.stdout
  serr = sys.stderr
  f = open(os.devnull, 'w')
  sys.stdout = f
  sys.stderr = f
  try:
    _create_unverified_https_context = ssl._create_unverified_context
  except AttributeError:
    pass
  else:
    ssl._create_default_https_context = _create_unverified_https_context

  nltk.download('words')
  nltk.download('wordnet')
  nltk.download('stopwords')
  #nltk.download('punct')

  # turn output off - too noisy
  sys.stdout = sout
  sys.stderr = serr

ensure_nlk_downloads()
