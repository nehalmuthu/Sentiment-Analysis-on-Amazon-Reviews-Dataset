'''
Version
Python 3.11.0
'''

import pandas as pd
import numpy as np
import nltk
#nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np

#Read Data
data = pd.read_csv('./data.tsv', sep='\t', on_bad_lines='skip',low_memory=False)

#Keep Reviews and Ratings

reviews=data[["review_body","star_rating"]].copy()
reviews = reviews[reviews["review_body"].notna()]

# We form three classes and select 20000 reviews randomly from each class.
reviews["star_rating"]=reviews["star_rating"].replace('1',1)
reviews["star_rating"]=reviews["star_rating"].replace(2,1)
reviews["star_rating"]=reviews["star_rating"].replace('2',1)
reviews["star_rating"]=reviews["star_rating"].replace('3',2)
reviews["star_rating"]=reviews["star_rating"].replace(4,3)
reviews["star_rating"]=reviews["star_rating"].replace('4',3)
reviews["star_rating"]=reviews["star_rating"].replace(5,3)
reviews["star_rating"]=reviews["star_rating"].replace('5',3)


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
class1_df = reviews[reviews["star_rating"]==1]
sample1=class1_df.sample(n = 20000,random_state=47)
sample1 = sample1.reset_index(drop=True)
class2_df=reviews[reviews["star_rating"]==2]
sample2=class2_df.sample(n = 20000,random_state=47)
sample2 = sample2.reset_index(drop=True)
class3_df = reviews[reviews["star_rating"]==3]
sample3=class3_df.sample(n = 20000,random_state=47)
sample3 = sample3.reset_index(drop=True)

reviews_df=pd.concat([sample1,sample2,sample3],axis=0,ignore_index=True)

len_before_cleaning=reviews_df['review_body'].str.len().mean()


# Data Cleaning

# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/


# lower case
reviews_df['review_body']=reviews_df['review_body'].str.lower()
#remove html tags
reviews_df['review_body'] = reviews_df['review_body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
# remove url
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: re.sub(r'http\S+','', str(x)))
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: re.sub(r'https\S+','', str(x)))
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: re.sub(r'www\.\S+','', str(x)))
# keeping only alphabets
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: re.sub(r'[^a-z A-Z]+',' ', str(x)))
# removing extra spaces
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
# strping spaces at start and end
reviews_df['review_body']=reviews_df['review_body'].str.strip()
# contractions
reviews_df['review_body'] =  reviews_df['review_body'].apply(lambda x: contractions.fix(x))

len_after_cleaning=reviews_df['review_body'].str.len().mean()

print(int(len_before_cleaning),",",int(len_after_cleaning))

len_before_preprocess=len_after_cleaning



# Pre-processing
# remove the stop words 
reviews_df['tokenized'] = reviews_df['review_body'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
reviews_df['stopwords_removed'] = reviews_df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

# perform lemmatization
reviews_df['pos_tags'] = reviews_df['stopwords_removed'].apply(nltk.tag.pos_tag)

#https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

reviews_df['wordnet_pos'] = reviews_df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
wnl = WordNetLemmatizer()
reviews_df['lemmatized'] = reviews_df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
reviews_df=reviews_df[["review_body","star_rating","lemmatized"]]
reviews_df["reviews_processed"]=reviews_df['lemmatized'].apply(lambda x: ' '.join(x))

len_after_preprocess = reviews_df['reviews_processed'].str.len().mean()

print(int(len_before_preprocess),",",int(len_after_preprocess))




# TF-IDF Feature Extraction - Running with No stop word removal and No Lemmatization
reviews_df=reviews_df[["reviews_processed","star_rating"]]
x_train ,x_test,y_train,y_test=train_test_split(reviews_df["reviews_processed"],reviews_df["star_rating"],
                                                test_size=0.2 ,
                                                random_state=15,stratify=reviews_df['star_rating'])

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
tfIdfVectorizer=TfidfVectorizer(use_idf=True,ngram_range=(1, 3),sublinear_tf=True)

# https://stats.stackexchange.com/questions/154660/tfidfvectorizer-should-it-be-used-on-train-only-or-traintest
train_tfIdf = tfIdfVectorizer.fit_transform(x_train)
test_tfIdf = tfIdfVectorizer.transform(x_test)


# Modelling

# https://stackoverflow.com/questions/48417867/access-to-numbers-in-classification-report-sklearn
def printMetrics(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    print("{:0.4f}".format(report['accuracy']))

    #print("{:0.4f}".format(report['1']['precision']),",","{:0.4f}".format(report['1']['recall']),",","{:0.4f}".format(report['1']['f1-score']))
    #print("{:0.4f}".format(report['2']['precision']),",","{:0.4f}".format(report['2']['recall']),",","{:0.4f}".format(report['2']['f1-score']))
    #print("{:0.4f}".format(report['3']['precision']),",","{:0.4f}".format(report['3']['recall']),",","{:0.4f}".format(report['3']['f1-score']))
    #print("{:0.4f}".format(report['macro avg']['precision']),",","{:0.4f}".format(report['macro avg']['recall']),",","{:0.4f}".format(report['macro avg']['f1-score']))


# Perceptron
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(train_tfIdf,y_train)
pred=model.predict(test_tfIdf)
print("Perceptron Accuracy")
printMetrics(y_test, pred)

# SVM
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=0, tol=1e-4)
model.fit(train_tfIdf,y_train)
pred=model.predict(test_tfIdf)
print("SVM Accuracy")
printMetrics(y_test, pred)


'''
 Logistic Regression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(train_tfIdf,y_train)
pred=model.predict(test_tfIdf)
printMetrics(y_test, pred)



Multinomial Naive Bayes
 https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(train_tfIdf,y_train)
pred=model.predict(test_tfIdf)

printMetrics(y_test, pred)

'''