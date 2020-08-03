#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt

#Import the training and testing data
train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")

#Initial feature selection
list_drop = ["num_outbound_cmds"]
train.drop(list_drop, axis = 1, inplace = True)
test.drop(list_drop, axis = 1, inplace = True)

#We need to handle categorical data in our dataset as they are also important
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
train1 = train.iloc[:, 1:2]
train2 = train.iloc[:, 2:3]
train3 = train.iloc[:, 3:4]
train4 = train.iloc[:, 40:41]

enc1 = OneHotEncoder(handle_unknown="ignore")
encc1 = pd.DataFrame(enc1.fit_transform(train1).toarray())

enc2 = OneHotEncoder(handle_unknown="ignore")
encc2 = pd.DataFrame(enc2.fit_transform(train2).toarray())

enc3 = OneHotEncoder(handle_unknown="ignore")
encc3 = pd.DataFrame(enc3.fit_transform(train3).toarray())

enc4 = OneHotEncoder(handle_unknown="ignore")
encc4 = pd.DataFrame(enc4.fit_transform(train4).toarray())

test1 = test.iloc[:,1:2]
test2 = test.iloc[:,2:3]
test3 = test.iloc[:,3:4]

enc11 = OneHotEncoder(handle_unknown="ignore")
encc11 = pd.DataFrame(enc11.fit_transform(test1).toarray())

enc22 = OneHotEncoder(handle_unknown="ignore")
encc22 = pd.DataFrame(enc22.fit_transform(test2).toarray())

enc33 = OneHotEncoder(handle_unknown="ignore")
encc33 = pd.DataFrame(enc33.fit_transform(test3).toarray())

objects = train.select_dtypes(include=['float64','int64'])
train_x = pd.concat([encc1,encc2,encc3,objects],axis=1)
train_y = encc4
train_x.head(10)

objects2 = test.select_dtypes(include=['float64','int64'])
test_x = pd.concat([encc11,encc22,encc33,objects2],axis=1)
test_x.head(10)

train_x = pd.DataFrame(train_x)
test_x = pd.DataFrame(test_x)

cols = train_x.columns
cols2 = test_x.columns

#Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# extract numerical attributes and scale it to have zero mean and unit variance
train_x = scaler.fit_transform(train_x)
train_x= pd.DataFrame(train_x, columns = cols)
test_x = scaler.fit_transform(test_x)
test_x= pd.DataFrame(test_x, columns = cols2)

#Model

