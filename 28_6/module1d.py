import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

tokenizer=RegexpTokenizer('[a-zA-Z]+')
stopwords=set(stopwords.words('english'))
ps=PorterStemmer()
def clean_review(review):
    review=review.lower()
    review=review.replace("<br><\br>"," ")
    tokens= tokenizer.tokenize(review)
    new_tokens=[ps.stem(token) for token in tokens if token not in stopwords]

    return " ".join(new_tokens)

import numpy as np
import matplotlib.pyplot as plt
data=open(r".\IMDB\imdb_trainX.txt","r",encoding="utf8")
x=data.readlines()
#print(x[1])

datum=open(r".\IMDB\imdb_testX.txt","r",encoding="utf8")
x_test=datum.readlines()




y=open(r".\IMDB\imdb_trainY.txt","r",encoding="utf8")
y=y.readlines()

y_test=open(r".\IMDB\imdb_testY.txt","r",encoding="utf8")
y_test=y_test.readlines()

x_cleaned=[clean_review(sent) for sent in x ]

x_test_cleaned=[clean_review(sent) for sent in x_test ]

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(ngram_range=(1,1))

cv=CountVectorizer(ngram_range=(1,1))

x_vect=cv.fit_transform(x_cleaned).toarray()

print(x_vect)

x_vect_test=cv.transform(x_test_cleaned).toarray()


from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x_vect,y)

arr=mnb.predict(x_vect_test)

arr=pd.DataFrame(data=arr,columns=["out"])

arr.to_csv('result.csv',index=False)


