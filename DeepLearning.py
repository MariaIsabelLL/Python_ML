import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import itertools
import os
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

###################################################################################
#Red Neuronal Simple
###################################################################################
# cargamos datos de pruebas, que cumplan lo siguiente: «-6 > x1 > 6» y «-6 > x2 > 6» será 1,
#caso contrario será 0.

def modelo_Simple():
    training_data = np.array([[7,8],[1,1],[1,6],[9,10],[4,5],[6,6],[1,3],[9,7]], "float32")
    target_data = np.array([[0],[1],[1],[0],[1],[0],[1],[1]], "float32")
     
    model = Sequential()
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
     
    model.fit(training_data, target_data, epochs=1000)
     
    # evaluamos el modelo
    scores = model.evaluate(training_data, target_data) 
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print (model.predict(training_data).round())

modelo_Simple()
    
def guardar_modelo(model):
    # serializar el modelo a JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serializar los pesos a HDF5
    model.save_weights("model.h5")
    print("Modelo Guardado!")
 
def cargar_modelo(): 
    # cargar json y crear el modelo
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargar pesos al nuevo modelo
    loaded_model.load_weights("model.h5")
    print("Cargado modelo desde disco.") 
    # Compilar modelo cargado y listo para usar.
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

###################################################################################
#Red Neuronal Recurrente Simple
###################################################################################
    
def graficos(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    
def modelo_SimpleRNN():
    
    max_features = 10000 #tamaño del vocabulario de imdb dataset
    maxlen = 500         #tamaño del vector cuando se realiza el padding
    embedding_size = 32  #tamaño del espacio vectorial donde se realizará el embedding

    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)
    print(y_train)
    print(input_train)
    
    model = Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embedding_size)) #max_feature=10,000 so, 320,000
    model.add(tf.keras.layers.SimpleRNN(32))               #(32+32+1)*32=2080
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
    
    history = model.fit(input_train, y_train,epochs=10, batch_size=32, validation_split=0.2)
    graficos(history)

modelo_SimpleRNN()

    
###################################################################################
#Red Neuronal Recurrente LSTM
###################################################################################
  
def modelo_LSTM():
    
    max_features = 10000 #tamaño del vocabulario de imdb dataset
    maxlen = 500         #tamaño del vector cuando se realiza el padding
    embedding_size = 32  #tamaño del espacio vectorial donde se realizará el embedding

    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequences')
    print(len(input_test), 'test sequences')
    
    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)
    print(y_train)
    print(input_train)
    
    model = Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embedding_size)) #max_feature=10,000 so, 320,000
    model.add(tf.keras.layers.LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
    
    history = model.fit(input_train, y_train,epochs=10, batch_size=32, validation_split=0.2)
    graficos(history)

modelo_LSTM()
