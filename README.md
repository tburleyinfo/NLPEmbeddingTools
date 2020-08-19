INSTALL PYTHON DEPENDENCIES
`python3 -m pip install -r requirements.txt`
`Insight: https://medium.com/@boscacci/why-and-how-to-make-a-requirements-txt-f329c685181e`

DO NOT USE THIS COMMAND: 
`pip freeze > requirements.txt`

SETUP CORENLP: https://stanfordnlp.github.io/CoreNLP/download.html
`git clone https://github.com/stanfordnlp/CoreNLP.git`

`brew install ant`

```
cd CoreNLP

sudo ant jar
```

`curl -O -L http://nlp.stanford.edu/software/stanford-corenlp-models-current.jar`

If you’ll be using CoreNLP frequently, the below lines are useful to have in your ~/.bashrc (or equivalent) file, replacing the directory /path/to/corenlp/ with the appropriate path to where you unzipped CoreNLP (3 replacements):

```
vim ~/.bashrc

{paste the following}
export CLASSPATH="$CLASSPATH:/path/to/corenlp/javanlp-core.jar:/path/to/corenlp/stanford-corenlp-models-current.jar";
for file in `find /path/to/corenlp/lib -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
```

To Run CoreNLP: 
`java -mx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat json -file input.txt`

To Start coreNLP Server: 
`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`




