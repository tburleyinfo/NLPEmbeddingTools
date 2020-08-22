

from NLPembeddingTools import PreProcess 
if __name__ == "__main__": 
    import os
    home = os.getcwd()
    df = PreProcess.metaDataFrame('./samples/', to_csv=True)
    os.chdir(home)
    print(df.head)
    

