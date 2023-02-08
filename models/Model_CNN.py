import os
import numpy as np
from Model import Model
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

class Model_CNN(Model) :
    _model_name : str = "CNN"
    _optimizer = "adam"
    _loss = "categorical_crossentropy"
    _metrics = 'accuracy'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = self._build_model()
        

    def _build_model(self,load=False):
        print("Building CNN model")
        if load:
            return self.load()
        else:
            model = keras.Sequential()
            model.add(Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(64, (3,3), activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(128, (3,3), activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(128, (3,3), activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Flatten())
            model.add(Dense(512, activation="relu"))
            model.add(Dense(2, activation="sigmoid"))

            print("CNN model created")
            model.compile(optimizer="adam",loss="binary_crossentropy", metrics=['accuracy'])
            model.summary()
            return model
