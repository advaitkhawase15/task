#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#Using the Digital recognizer dataset(for Training)
dataset=pd.read_csv("train.csv")

#Dividing dataset into x_train & y_train
y=dataset['label']
x=dataset.drop('label', axis=1)

#Converting pandas to numpy
x=x.values
y=y.values

#Knowning the number of entries
train_entries=x.shape[0]

#Converting data in 4-D
x=x.reshape(train_entries, 28, 28 , 1)

#Spliting data into traning and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#Changing the data type of x_train and x_test
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#coverting y_train & y_test into categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[2]:


from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Dense,Flatten
import keras

#Making the model
model=Sequential()
model.add(Convolution2D(filters=32,
                        kernel_size=(8,8),
                        activation='relu',
                        input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(units=40,
                activation='relu'))
model.add(Dense(units=10,
                activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
model.summary()


# In[3]:


#Fiting the datset in model
model.fit(x_train,y_train,epochs=1,verbose=1)


# In[4]:


score=model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

