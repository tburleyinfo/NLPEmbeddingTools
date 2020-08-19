# Divisive.py: First unsupervised learning solution for noisy trigger problem. 


# Step Two: Vectorization
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

#also test the vectorization in the SVM to see if it works 

# Structural Embedding of Syntactic Trees for Machine Comprehension: https://arxiv.org/abs/1703.00572
# SENSE2VEC - A FAST AND ACCURATE METHOD FOR WORD SENSE DISAMBIGUATION IN NEURAL WORD EMBEDDINGS: https://arxiv.org/pdf/1511.06388.pdf 
# https://www.quora.com/Is-there-a-word-embedding-scheme-that-is-part-of-speech-tag-specific
# https://explosion.ai/blog/sense2vec-with-spacy
# https://onlinelibrary.wiley.com/doi/10.1111/exsy.12460
# https://towardsdatascience.com/a-deeper-look-into-embeddings-a-linguistic-approach-89cc428a29e7


#Main
from NLPembeddingTools import PreProcess 
if __name__ == "__main__": 
    import os
    home = os.getcwd()
    data = PreProcess.textract('./samples/')
    os.chdir(home)

from NLPembeddingTools import Vectorize 
model = Vectorize.word2vec(data[0])


#from NLPembeddingTools import Reshape
#model = Reshape.w2v_scikit(data[0])


#from NLPembeddingTools import Classify
#Classify.support_vector_machine(model, data[0])


from NLPembeddingTools import Visualize as vis 
vis.ScatterPlot(model)







