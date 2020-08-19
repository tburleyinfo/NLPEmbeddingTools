#ELMo: 

#https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/

"""
Unlike traditional word embeddings such as word2vec and GLoVe, the ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts.

I can imagine you asking – how does knowing that help me deal with NLP problems? Let me explain this using an example.

Suppose we have a couple of sentences:

I read the book yesterday.
Can you read the letter now?
Take a moment to ponder the difference between these two. The verb “read” in the first sentence is in the past tense. And the same verb transforms into present tense in the second sentence. This is a case of Polysemy wherein a word could have multiple meanings or senses.
"""


from NLPembeddingTools import PreProcess 
if __name__ == "__main__": 
    data = PreProcess.textract('./samples/')


#Tutorial: https://towardsdatascience.com/transfer-learning-using-elmo-embedding-c4a7e415103c


from NLPembeddingTools import Vectorize as vect 
vect.elmo(data[0])


#https://github.com/plasticityai/magnitude/blob/master/ELMo.md
#So Apparently the tensors returned from ELMo embeddings are to complex to return vocabularies. 
#So we can't visualize them. We can however, unroll and reshape them to fit into a different classifier or clustering algorithm. 

elmo_vecs = Magnitude('elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude')
sentence  = elmo_vecs.query(["play", "some", "music", "on", "the", "living", "room", "speakers", "."])
# Returns: an array of size (9 (number of words) x 768 (3 ELMo components concatenated))
unrolled = elmo_vecs.unroll(sentence)




