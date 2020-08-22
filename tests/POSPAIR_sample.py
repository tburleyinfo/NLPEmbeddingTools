
from NLPembeddingTools import PreProcess 
if __name__ == "__main__": 
    data = PreProcess.textract('./samples/')


# Make Sure You Run the CoreNLP Server to tag everything. 
# Go to /Users/timothyburley/Downloads/stanford-corenlp-4.1.0
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
from pycorenlp import StanfordCoreNLP
StanfordCoreNLP('http://localhost:9000')

from NLPembeddingTools import Vectorize as vect 
from NLPembeddingTools import Visualize as vis 

vis.ScatterPlot(vect.pospair(data[4]))








