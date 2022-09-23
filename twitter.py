import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

d = pd.read_csv(r"C:\Users\sonas\Downloads\twitterdataset(1).csv")
d = d.drop(labels=["id","label"],axis=1)
d.head()

def remo(inp,pattern):
    r = re.findall(pattern, inp)
    for i in r:
        inp = re.sub(i, "", inp)
    return inp
d['new'] = np.vectorize(remo)(d['tweet'], "@[\w]*")
d.head()

d['new'] = d['new'].apply(lambda x:' '.join([w for w in x.split() if '.com' not in w]))
d.head()

d['new'] = d['new'].str.replace("[^a-zA-Z]", " ")
d.head()

import nltk
nltk.download('punkt')
t=pd.DataFrame()

from nltk.tokenize import word_tokenize
t['tokens'] = d['new'].apply(lambda x: word_tokenize(x.lower()))
t.head()

from nltk.stem import PorterStemmer
st = PorterStemmer()
t['tokens']=t['tokens'].apply(lambda x: [st.stem(i) for i in x])
t.head()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
s_w = set(stopwords.words('english'))
t['tokens'] = t['tokens'].apply(lambda x: [i for i in x if(i not in s_w)])
t.head()

t['tokens'] = t['tokens'].apply(lambda x:' '.join([w for w in x if len(w)>3]))
t.head()

t = t.replace('',np.NaN)
t.dropna(axis=0,inplace=True)
t.head()
tokens=[]
for i in list(t.loc[:,'tokens']):
    tokens += word_tokenize(i)
print(tokens)

mpw = []
for i in set(tokens):
    if(tokens.count(i)>500):
        mpw.append(i)
        print(i,tokens.count(i))

        
 
plt.bar(mpw,[tokens.count(i) for i in mpw])
