Best practises for solving ML Problem

1)DataPreprocessing and Cleaning ---->feature engineering

2)Splitting the data into training and testing sets ------->train test split

3)BOW or TF_IDF ------->any text to vector conversion techniques

4)training ml model

import pandas as pd 
df=pd.read_csv('smsspamcollection.csv',sep="\t",names=["label","message"])
df.head( )
label	message
0	ham	Go until jurong point, crazy.. Available only ...
1	ham	Ok lar... Joking wif u oni...
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
3	ham	U dun say so early hor... U c already then say...
4	ham	Nah I don't think he goes to usf, he lives aro...
import re 
import nltk
from nltk.corpus import stopwords
# Data cleaning and preprocessing
# Remove comments for print and break to check how each line of code is working
corpus=[]
for i in range(len(df)):
    #print(df['message'][i])
    review=re.sub('[^a-zA-Z]',' ',df["message"][i]) # replace everything except alphabets
    #print(review)
    review=review.lower() # converting into lower case 
    #print(review)
    review=review.split() # splitting into words
    #print(review)
    review=" ".join(review) # Joining together 
    #print(review)
    corpus.append(review) # Appending cleaned sentence to corpus
    #print(corpus)
    #break
#corpus
Output Variable

y=pd.get_dummies(df["label"])
y=y.iloc[:,0].values
#y
array([ True,  True, False, ...,  True,  True,  True])
Applying Train Test Split

print(type(corpus))
print(len(corpus))
<class 'list'>
5571
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(corpus,y,test_size=0.20)
#print(X_train[2])
#print(len(X_train))
#print(len(y_train))
ok i am a gentleman and will treat you with dignity and respect
4456
4456
Lets apply Word Embedding Techniques to convert text into vectors

# Lets apply Bag of Words 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500,ngram_range=(1,2))
#Independent features 
X_train=cv.fit_transform(X_train).toarray()
X_test=cv.transform(X_test).toarray()

Applying Ml algorithm

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB()
spam_detect_model.fit(X_train,y_train)
MultinomialNB()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# Predictions 
y_pred=spam_detect_model.predict(X_test)
Evaluating the Model

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(f"accuracy_score :{accuracy_score(y_test,y_pred)}")
print(f"confusion_matrix :\n{confusion_matrix(y_test,y_pred)}")
print("\n Classification report")
print(classification_report(y_test,y_pred))
accuracy_score :0.9838565022421525
confusion_matrix :
[[141  10]
 [  8 956]]

 Classification report
              precision    recall  f1-score   support

       False       0.95      0.93      0.94       151
        True       0.99      0.99      0.99       964

    accuracy                           0.98      1115
   macro avg       0.97      0.96      0.97      1115
weighted avg       0.98      0.98      0.98      1115

Crating the TF-IDF Model

import numpy
# Train Test split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(corpus,y,test_size=0.20)
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf=TfidfVectorizer(max_features=2500,ngram_range=(1,2))
#Independent features
X_train=tf_idf.fit_transform(X_train).toarray()
X_test=tf_idf.transform(X_test).toarray()
Applying ml algorithm

from sklearn.naive_bayes import MultinomialNB
spam_model=MultinomialNB()
spam_model.fit(X_train,y_train)
MultinomialNB()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# Predictions for TF_IDF model
y_pred=spam_model.predict(X_test)
Evaluating the TF-IDF model

print(f"accuracy_score :{accuracy_score(y_test,y_pred)}")
print(f"confusion_matrix :\n{confusion_matrix(y_test,y_pred)}")
print("\n Classification report")
print(classification_report(y_test,y_pred))
accuracy_score :0.9838565022421525
confusion_matrix :
[[139  18]
 [  0 958]]

 Classification report
              precision    recall  f1-score   support

       False       1.00      0.89      0.94       157
        True       0.98      1.00      0.99       958

    accuracy                           0.98      1115
   macro avg       0.99      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115

 