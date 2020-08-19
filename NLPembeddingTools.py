# Step one: Data Loading

#TF-IDF decreases accuracy of Neural Embeddings
# Heres a comparitive analysis: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/


import json
from bs4 import BeautifulSoup
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os

#ELMo is a TensorFlow Algorithm. 
import tensorflow_hub as hub
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()

class PreProcess:
    def textract(directory):

        def jsonReader(filename):
            with open(filename) as f_in:
                return(json.load(f_in))

        textlist = []
        home = os.getcwd()
        os.chdir(directory)
        data = []
        for filename in os.listdir(os.getcwd()):
            name, file_extension = os.path.splitext(filename)
            if '.json' in file_extension:
                my_data = jsonReader(filename)
                data.append(my_data)

        from cleantext import clean
        
        for article in data:
            soup = json.dumps(article["Document"]["Content"], indent=4, sort_keys=True)
            soup = BeautifulSoup(soup, features="lxml")
            #print(soup.prettify())
            doc = []
            clean_doc = []
            for line in soup.find("bodytext").findAll("p", recursive=False): 
                doc.append(line.getText())
            for text in doc:
                cleaned = clean(text,
                            fix_unicode=True,               # fix various unicode errors
                            to_ascii=True,                  # transliterate to closest ASCII representation
                            lower=False,                    # lowercase text
                            no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                            no_urls=False,                  # replace all URLs with a special token
                            no_emails=False,                # replace all email addresses with a special token
                            no_phone_numbers=False,         # replace all phone numbers with a special token
                            no_numbers=False,               # replace all numbers with a special token
                            no_digits=False,                # replace all digits with a special token
                            no_currency_symbols=False,      # replace all currency symbols with a special token
                            no_punct=False,                 # fully remove punctuation
                            replace_with_url="<URL>",
                            replace_with_email="<EMAIL>",
                            replace_with_phone_number="<PHONE>",
                            replace_with_number="<NUMBER>",
                            replace_with_digit="0",
                            replace_with_currency_symbol="<CUR>",
                            lang="en"                       # set to 'de' for German special handling
                        )
                clean_doc.append(cleaned)
            textlist.append(clean_doc)
        os.chdir(home)
        return textlist


class Reshape: 
    #Tutorial: https://www.youtube.com/watch?v=fCVuiW9AFzY
    def w2v_scikit(data): 
        from gensim.test.utils import common_texts
        from gensim.sklearn_api import W2VTransformer
        
        # Create a model to represent each word by a 10 dimensional vector.
        model = W2VTransformer(size=len(data), min_count=1, seed=1)
        #print(model.gensim_model.wv.vocab)
        
        # What is the vector representation of the word 'graph'?
        wordvecs = model.fit(data).transform(['taken', 'arms'])
        assert wordvecs.shape == (len(data))
        return wordvecs

    def reshape(model): 
        pass


class Visualize: 
    def ScatterPlot(model): 
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        #https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        # Principal Component Analysis or PCA.
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        plt.scatter(result[:, 0], result[:, 1])
        words = list(model.wv.vocab)

        for i, word in enumerate(words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))

        from matplotlib.pyplot import figure
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        #fig.savefig('test3png.png', dpi=100)
        plt.show()

    def Dendrite(model): 
        """
        from scipy.cluster.hierarchy import dendrogram, linkage

        l = linkage(model.wv.syn0, method='complete', metric='seuclidean')

        # calculate full dendrogram
        plt.figure(figsize=(200, 200)) 
        plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('word')
        plt.xlabel('distance')

        dendrogram(
            l,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=3.,  # font size for the x axis labels
            orientation='top',
            leaf_label_func=lambda v: str(model.wv.index2word[v])
        )
        plt.show()
        """
        pass


class Vectorize:
    def elmo(data): #Input should be a list of sentences. 
        #Use this to define a function for creating tf sessions so it's less resource heavy
        def embed_elmo2(module):
            with tf.Graph().as_default():
                sentences = tf.placeholder(tf.string)
                embed = hub.Module(module)
                embeddings = embed(sentences)
                session = tf.train.MonitoredSession()
            return lambda x: session.run(embeddings, {sentences: x})

        embed_fn = embed_elmo2('module/module_elmo2')
        model = embed_fn(data).shape
        return model


    def bert(data): 
        from bert_embedding import BertEmbedding
        bert_embedding = BertEmbedding()
        result = bert_embedding(data[0])
        return result

    def pospair(data):
        #Recall you need to activate the CORENLP server to train POSpair. 
        import POSPair
        model = POSPair.POSPairWordEmbeddings(data)
        if model is None:
            print("Error: Nonetype returned. Make sure you are running a Server instance of Stanford CoreNLP. \n java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000")
            return None
        return model

    def word2vec(data): 
        from nltk.tokenize import word_tokenize, sent_tokenize
        import nltk
        from gensim.models import Word2Vec
        words = []
        for i in range(len(data)):
            #nltk.download('averaged_perceptron_tagger')
            text = word_tokenize(data[i])
            #word = nltk.pos_tag(text)
            words.append(text)

        #print(words)
        sample = []
        #samples = []
        for word in words:
            for entry in word:
                line = "".join(str(v) for v in entry)
                sample.append([line])
        
        model = Word2Vec(sample, min_count = 1, size = 100, window = 5)
        #print(model.wv.vocab)
        return model


class Cluster: 
    def dbscan(tensor):
        pass
    
    def AHC(tensor): 
        pass


class Classify: 
    #http://www.cs.unibo.it/~montesi/CBD/Articoli/2015_Support%20vector%20machines%20and%20Word2vec%20for%20text%20classification%20with%20semantic%20features.pdf
    def support_vector_machine(tensor, data): 
        def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])


        from sklearn import svm
        from sklearn.metrics import accuracy_score
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        print(len(tensor), len(data))
        """SVM.fit(tensor, data)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(data)
        # Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, data)*100)"""
        

    def RNN(tensor): 
        pass


class Coincedence: 
    def ECA(EventSeriesX, EventSeriesY, delT, tau=0, ts1=None, ts2=None):
        """
        Event coincidence analysis:
        Returns the precursor and trigger coincidence rates of two event series
        X and Y.
        :type EventSeriesX: 1D Numpy array
        :arg EventSeriesX: Event series containing '0's and '1's
        :type EventSeriesY: 1D Numpy array
        :arg EventSeriesY: Event series containing '0's and '1's
        :arg delT: coincidence interval width
        :arg int tau: lag parameter
        :rtype: list
        :return: [Precursor coincidence rate XY, Trigger coincidence rate XY,
            Precursor coincidence rate YX, Trigger coincidence rate YX]
        """

        # Count events that cannot be coincided due to tau and delT
        if not (tau == 0 and delT == 0):
            # Start of EventSeriesX
            n11 = np.count_nonzero(EventSeriesX[:tau+delT])
            # End of EventSeriesX
            n12 = np.count_nonzero(EventSeriesX[-(tau+delT):])
            # Start of EventSeriesY
            n21 = np.count_nonzero(EventSeriesY[:tau+delT])
            # End of EventSeriesY
            n22 = np.count_nonzero(EventSeriesY[-(tau+delT):])
        else:
            # Instantaneous coincidence
            n11, n12, n21, n22 = 0, 0, 0, 0
        # Get time indices
        if ts1 is None:
            e1 = np.where(EventSeriesX)[0]
        else:
            e1 = ts1[EventSeriesX]
        if ts2 is None:
            e2 = np.where(EventSeriesY)[0]
        else:
            e2 = ts2[EventSeriesY]
        del EventSeriesX, EventSeriesY, ts1, ts2
        # Number of events
        l1 = len(e1)
        l2 = len(e2)
        # Array of all interevent distances
        dst = (np.array([e1]*l2).T - np.array([e2]*l1))
        # Count coincidences with array slicing
        prec12 = np.count_nonzero(np.any(((dst - tau >= 0)
                                * (dst - tau <= delT))[n11:, :],
                                axis=1))
        trig12 = np.count_nonzero(np.any(((dst - tau >= 0)
                                * (dst - tau <= delT))
                                [:, :dst.shape[1]-n22],
                                axis=0))
        prec21 = np.count_nonzero(np.any(((-dst - tau >= 0)
                                * (-dst - tau <= delT))[:, n21:],
                                axis=0))
        trig21 = np.count_nonzero(np.any(((-dst - tau >= 0)
                                * (-dst - tau <= delT))
                                [:dst.shape[0]-n12, :],
                                axis=1))
        # Normalisation and output
        return (np.float32(prec12)/(l1-n11), np.float32(trig12)/(l2-n22),
                np.float32(prec21)/(l2-n21), np.float32(trig21)/(l1-n12))
                