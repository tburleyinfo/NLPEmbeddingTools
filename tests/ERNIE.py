

from NLPembeddingTools import PreProcess 
if __name__ == "__main__": 
    data = PreProcess.textract('./samples/')

from NLPembeddingTools import Vectorize as vect 

result = vect.bert(data[0]) #Returns a Really big tensor. 
print(result[0])