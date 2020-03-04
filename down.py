import ssl
import nltk

def ensure_nlk_downloads() :
    try:
      _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
      pass
    else:
      ssl._create_default_https_context = _create_unverified_https_context    
    nltk.download('wordnet')
    nltk.download('stopwords')
    print('')
ensure_nlk_downloads()

