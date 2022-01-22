import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the data from csv file to a pandas dataFrame
raw_mail_data = pd.read_csv("E:/Python/Flask/mail_data.csv")
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


#Label Encoding spam mail as 0 and ham mail as 0
mail_data.loc[mail_data['Category'] == 'spam','Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham','Category',] = 1

#seperating data as text and labels
X = mail_data['Message']
Y = mail_data['Category']


#splitting data into training and testing data
#Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

#Feature Extractions  using TFIDFVectorizer 
#transform the text data into feature vectors(numerical values) so that they can be used as input to logistic regression
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test value as integer

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#Training the model
#logistic regression

model = LogisticRegression()

model.fit(X_train_features,Y_train)
input_data_feature = feature_extraction.transform(input_mail)

#Evaluating the trained model
#Prediction on trainig data
# prediction_on_training_data = model.predict(X_train_features)

# accuracy_on_training_data = accuracy_score(Y_train,prediction_on_training_data)
# print(accuracy_on_training_data)


#Building a predictive System

# input_mail = ["I've bee working in this URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18school get money and become rich after clicking on this ads you will be getting money withing 10 seconds for so many days. I want you to figure out what is the wrong thing going on in our llie andassdiasn dasjdnap[p dnlksdap shjasdasd a"]







# import pickle

# with open('model_pickle.pkl','wb') as f:
#   pickle.dump(model,f)