#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing Dataset
dataset = pd.read_csv('/content/drive/My Drive/Colab Notebooks/SOM/Self_Organizing_Maps/Credit_Card_Applications.csv')
print(dataset.head())
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)
print(X)


#Training the SOM
#!pip install minisom
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#Visualizing the Results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
  w=som.winner(x)
  plot(w[0] + 0.5,
       w[1] + 0.5,
       markers[y[i]],
       markeredgecolor = colors[y[i]],
       markerfacecolor='None',
       markersize =10,
       markeredgewidth=2)
show()

#Finding the Frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(6,8)], mappings[(5,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)
print(frauds)

#Creation of Hybrid Deep Learning Model
#Going from Unsupervised to supervised

#Creating matrix of features
customers = dataset.iloc[:,1:].values

#Creating Dependent Variables
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1
print(is_fraud) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)



# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
#building ANN
clf = Sequential()
#adding first hidden layer
clf.add(Dense(output_dim=2,init = 'uniform',activation='relu',input_dim=15))
clf.add(Dropout(p=0.1))

clf.add(Dropout(p=0.1))
#adding output layer
clf.add(Dense(output_dim=1,init = 'uniform',activation='sigmoid'))

#compiling ANN
clf.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])


#fitting 
clf.fit(customers,is_fraud,batch_size=1,epochs=2)


# Predicting the Test set results
y_pred = clf.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]

print(y_pred)
