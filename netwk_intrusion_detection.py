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

#carry out Univariate Feature selection to check out the important features.
from sklearn.feature_selection import SelectKBest, f_classif, RFE
X = train_x.iloc[0:,0:-1] # Features columnsx = x.astype(str)
y = train_y # Label column
bestfeatures = SelectKBest(score_func = f_classif, k = "all")
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat the two dataframes for better visualization
features_scores=pd.concat([dfcolumns,dfscores], axis = 1)
features_scores.columns = ['features ', 'scores']
print(features_scores.nlargest(50,'scores'))

#Lets create the RFE model and select the best attributes to be used for further analysis later
rfe = RFE(model, n_features_to_select = 40)
rfe = rfe.fit(train_x,train_y)

feature_map = [(i,v) for i, v in itertools.zip_longest(rfe.get_support(), cols)]
selected_features = [v for i, v in feature_map if i == True]
print(selected_features)

#Performing Machine learning procedure on our dataset

#Split the data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,test_size = 0.3, random_state = 42)

#We test several machine learning algorithms to see whic gives us the best results
# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred, Y_test)
precision = precision_score(y_pred, Y_test)
recall = recall_score(y_pred, Y_test)

print("The accuracy gotten from linear regression model is", accuracy)
print("The Precision gotten from the linear regression model is", precision)
print("The recall score from the model is", recall)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(X_train, Y_train)

# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)

cm2 = confusion_matrix(Y_test, y_pred2)
print(cm2)

accuracy = accuracy_score(y_pred2, Y_test)
precision = precision_score(y_pred2, Y_test)
recall = recall_score(y_pred2, Y_test)

print("The accuracy for the decision tree classifier is", accuracy)
print("The precision using the decision tree classifier is", precision)
print(" The recall score using the decision tree classifier is", recall)

#Using only the important features found earlier to create another set of models to see if we get better results
list_new = selected_features
train_x_new = train_x.loc[:, train_x.columns.intersection(list_new)]
test_x_new = train_x.loc[:, train_x.columns.intersection(list_new)]

from sklearn.model_selection import train_test_split
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(train_x_new,train_y,test_size = 0.3, random_state = 42)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier3.fit(X_train2, Y_train2)

# Predicting the Test set results
y_pred3 = classifier3.predict(X_test2)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(Y_test2, y_pred3)
print(cm3)

accuracy = accuracy_score(y_pred3, Y_test2)
precision = precision_score(y_pred3, Y_test2)
recall = recall_score(y_pred3, Y_test2)

print("The accuracy for the decision tree classifier is", accuracy)
print("The precision using the decision tree classifier is", precision)
print(" The recall score using the decision tree classifier is", recall)
