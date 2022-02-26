
** The system uses dependency links for building Text Graphs, that with help of a centrality algorithm like *PageRank*, extract relevant keyphrases, summaries and relations from text documents. A *SWI-Prolog* based module adds an interactive shell for talking about the document with a dialog agent that extracts for each query the most relevant sentences covering the document. Spoken dialog is also available if the OS supports it. Developed with *Python 3*, on OS X, but also working on Linux.**


# DeepRank is based on two packages.

# TextGraphCrafts

Python-based summary, keyphrase and relation extractor from text documents using dependency graphs.

HOME: https://github.com/ptarau/TextGraphCrafts

## Project Description

** The system uses dependency links for building Text Graphs, that with help of a centrality algorithm like *PageRank*, extract relevant keyphrases, summaries and relations from text documents.  Developed with *Python 3*, on OS X, but portable to Linux.**


## Dependencies:

- python 3.7 or newer, pip3, java 9.x or newer. Also, having git installed is recommended for easy updates
- ```pip3 install nltk```
-  also, run in python3 something like 


```
import nltk
nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')
```

- or, if that fails on a Mac, use run``` python3 down.py``` 
to collect the desired nltk resource files.
- ```pip3 install networkx```
- ```pip3 install requests```
- ```pip3 install graphviz```, also ensure .gv files can be viewed
- ```pip3 install stanfordnlp``` parser
- Note that ```stanfordnlp ``` requires torch binaries which are easier to instal with ````anaconda```.

Tested with the above on a Mac, with macOS Mojave and Catalina and on Ubuntu Linux 18.x.

## Running it:
#### in a shell window, run
 *start_server.sh*
#### in another shell window, start with

```python3 -i tests.py```

and then interactively, at the ">>>" prompt, try

```
>>> t1()
>>> t2()
>>> ...
>>> t9()
>>> t2()
>>> t0()
```

#### see how to activate other outputs in file 

```deepRank.py```

#### text file inputs (including the US Constitution const.txt) are in the folder

```examples/```

 
### Handling PDF documents

The easiest way to do this is to install *pdftotext*, which is part of [Poppler tools](https://poppler.freedesktop.org/).

If pdftotext is installed, you can place a file like *textrank.pdf*
already in subdirectory pdfs/ and try something similar to:

Change setting in file params.py to use the system with
other global parameter settings.

### Alternative NLP toolkit

*Optionally*, you can activate the alternative Stanford CoreNLP toolkit as follows:

- install [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and unzip in a derictory of your choice (ag., the local directory)
- edit if needed ```start_parser.sh``` with the location of the parser directory
- override the ```params``` class and set ```corenlp=True```

*Note however that the Stanford CoreNLP is GPL-licensed, which can place restrictions on proprietary software activating this option.*


## Project Description

** The system uses package ```text_graph_crafts``` based on dependency links for building Text Graphs, that with help of a centrality algorithm like *PageRank*, extract relevant keyphrases, summaries and relations from text documents. 

A *SWI-Prolog* based module adds an interactive shell for talking about the document with a dialog agent that extracts for each query the most relevant sentences covering the document. Spoken dialog is also available if the OS supports it. Developed with *Python 3*, on OS X, but portable to Linux.**


## Dependencies:

- python 3.7 or newer, pip3, java 9.x or newer, SWI-Prolog 8.x or newer, graphviz
- also, having git installed is recommended for easy updates
- ```pip3 install text_graph_crafts```

#### see how to activate other outputs in file 

```https://github.com/ptarau/TextGraphCrafts/blob/master/text_graph_crafts/deepRank.py```

The second is activated with
 
 ```python3 -i qpro.py```
 
or the shorthand script ```qgo```.
 
It requires SWI-Prolog to be installed and available in the path as the executable ```swipl``` and the Python to Prolog interface ```pyswip```, to be installed with

```pip3 install pyswip```
 
Note that on MacOS you will need to do something like:

export DYLD_FALLBACK_LIBRARY_PATH=/Applications/SWI-Prolog.app/Contents/Frameworks/

to help pyswip find your SWI-Prolog library.

It activates a Prolog process to which Python sends interactively queries about a selected document. Answers are computed by Prolog and then, if the parameter ```quiet``` is off, spoken using the ```say``` OS-level facility (available on OS X and Linux machines.

Prolog relation files, generated on the Python side are associated to each document as well as the queries about it. They are stored in the same directory as the document.

Try
```
>>> t1() 
...
>>> t9()
>>> t0()

or

>>> chat('const')

to interactively chat about the US Constitution. The same
for other documents in the examples folder.

### Handling PDF documents

The easiest way to do this is to install *pdftotext*, which is part of [Poppler tools](https://poppler.freedesktop.org/).

If pdftotext is installed, you can place a file like *textrank.pdf*
already in subdirectory pdfs/ and try something similar to:

```
>>> pdf_chat('textrank')
```
which activates a dialog about the TextRank paper. Also

```
>>> pdf_chat('logrank')
```
activates a dialog about *pdfs/logrank.pdf*, which describes
the architecture of the current system.

Change setting in file params.py to use the system with
other global parameter settings.


