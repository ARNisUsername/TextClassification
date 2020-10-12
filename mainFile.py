import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

bbc = pd.read_csv('bbc-text.csv')

X = bbc['text']
y = bbc['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Filtering and tokenizing of stopwords(useless words)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)

#Make long and short documents share same info and weigh down common words
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#Train the model using MultinomialNB and the updated X_train
clf = MultinomialNB().fit(X_train_tf, y)

#Make function to change a string into one the model can predict
def changeString(theStr):
    docs = [theStr]
    X_new_counts = count_vect.transform(docs)
    return tf_transformer.transform(X_new_counts)

print(clf.predict(changeString("The Los Angeles Lakers pay tribute to Kobe Bryant - who died in a helicopter crash in January - after winning their first NBA title in a decade.")))
