import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score, f1_score
import pickle

data = pd.read_csv('IMDB-Dataset.csv')
# data.info()
data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)
# print(data.head(10))

# Clean reviews
def clean(text):
  # Remove HTML tags
  tags = re.compile(r'<.*?>')
  text = re.sub(tags, '', text)

  # remove all non-alphanumeric characters
  non_alphanum = re.compile(r'\W_+') # '\W = [^a-zA-Z0-9_]
  text = re.sub(non_alphanum, '', text)

  # remove stopwords
  stop_words = set(stopwords.words('english'))
  text = " ".join([word for word in text.split() if word not in stop_words])

  # stem words
  ss = SnowballStemmer('english')
  text = "".join([ss.stem(word) for word in text])

  # convert to lowercase
  return text.lower()

data.review = data.review.apply(clean)
print(data.head(10))

# bag of words model
X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()
print("X.shape = ",X.shape)
print("y.shape = ",y.shape)
print(X)

# train test split
trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=9)
print("Train shapes : X = {}, y = {}".format(trainx.shape,trainy.shape))
print("Test shapes : X = {}, y = {}".format(testx.shape,testy.shape))

# fit and train models
gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

# display prediction metrics for each model
ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

# accuracy score
print("Prediction/Accuracy Metrics")
print("Gaussian Accuracy = ", accuracy_score(testy,ypg))
print("Multinomial Accuracy= ", accuracy_score(testy,ypm))
print("Bernoulli Accuracy= ", accuracy_score(testy,ypb))

# f1 score
print("Gaussian F1 score = ", f1_score(ypg, testy, average="weighted"))
print("Multinomial F1 score = ", f1_score(ypm, testy, average="weighted"))
print("Bernoulli F1 score = ", f1_score(ypb, testy, average="weighted"))