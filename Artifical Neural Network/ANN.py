# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"/content/drive/My Drive/Colab Notebooks/Churn_modelling.csv")
X = dataset.iloc[:,3:13 ].values
y = dataset.iloc[:, 13].values


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')

X = ct.fit_transform(X)
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
#building ANN
clf = Sequential()
#adding first hidden layer
clf.add(Dense(output_dim=6,init = 'uniform',activation='relu',input_dim=11))
clf.add(Dropout(p=0.1))
#adding second hidden layer
clf.add(Dense(output_dim=6,init = 'uniform',activation='relu'))
clf.add(Dropout(p=0.1))
#adding output layer
clf.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))

#compiling ANN
clf.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN
clf.fit(X_train,y_train,batch_size=10,nb_epoch=100)





# Predicting the Test set results
y_pred = clf.predict(X_test)
y_pred = (y_pred>0.5)
print(y_pred)


#predicting
#geo:France
#credit score:600
#gender;male
#Age:40
#bal:60000
#no of prod:2
#has cred card:Yes
#is act mem : yes
#est sal : 50000
new_prediction = clf.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction>0.5)
print(new_prediction)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
  
  clf = Sequential()
  clf.add(Dense(output_dim=6,init = 'uniform',activation='relu',input_dim=11))

  clf.add(Dense(output_dim=6,init = 'uniform',activation='relu'))

  clf.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))

  clf.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

  return clf


classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
print(accuracies)


print(np.mean(accuracies))
print(np.std(accuracies))



#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
  
  clf = Sequential()
  clf.add(Dense(output_dim=6,init = 'uniform',activation='relu',input_dim=11))

  clf.add(Dense(output_dim=6,init = 'uniform',activation='relu'))

  clf.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))

  clf.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

  return clf
classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']
              }
gridSearch = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=10)
gridSearch = gridSearch.fit(X_train,y_train)



best_parameters = gridSearch.best_params_
best_accuracy=gridSearch.best_score_
print(best_parameters)
print(best_accuracy)
