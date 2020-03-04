from abc import ABC, abstractmethod

# nlp toolkit plugin - abstract class

class NLP_API(ABC):
    def __init__(self, text):
        self.text = text
        self.triples = None
        self.lemmas = None
        self.words = None
        self.tags=None

    @abstractmethod
    def get_triples(self):
        pass

    @abstractmethod
    def get_lemmas(self):
        pass

    @abstractmethod
    def get_words(self):
        pass

    @abstractmethod
    def get_tags(self):
      pass

    def get_all(self):
        return self.get_triples(), self.get_lemmas(), self.get_words(), self.get_tags()

