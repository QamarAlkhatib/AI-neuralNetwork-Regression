# CCPP --> Combined Cycle Power Plant in regression 

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_excel('Folds5x2_pp.xlsx') 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values #dependent variable to predict (last column EP)

# in case you want to see the output
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  
# this function that will take as input the data set but not in the form of the data frame, in the form of the two subset X, matrix features and y the dependent variable vector 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
#test_size we will take less than 2000 out of 9568 data observation in the data set
#random_state will fix the randomness in case to have exactly the same split of the training set and the test set

# Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Dense is for the connection between the input layer and the first hidden layer 


### Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))

#Training the ANN
#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Adam is the most popular optimizer when we want to use the gradient descent algo and it will reduse the loss between the predictions and real results

#Training the ANN model on the Training set
ann.fit(x_train, y_train, batch_size = 32 ,epochs = 100)

#Predicting the results of the Test set
y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
 