import os
import numpy as np
from Model import Model
import tensorflow as tf

class Model_FT_ResNet50(Model) :
    _model_name : str = "ResNet50"
    _nb_layers_to_learn : int = 2
    _optimizer = tf.keras.optimizers.RMSprop(lr=1e-5)
    _loss = "categorical_crossentropy"
    _metrics = 'accuracy'

    def __init__(self, *args, **kwargs):
        self.nb_layers_to_learn = kwargs.pop('nb_layers_to_learn')
        super().__init__(*args, **kwargs)
        self._model = self._build_model()
        

    def _build_model(self,load=False):
        print("Building model fine tuning")
        if load:
            return self.load()
        else:

            model_base = tf.keras.applications.resnet50.ResNet50(include_top=False , weights="imagenet", input_tensor=tf.keras.Input(shape=(224, 224, 3)))
            
            model_base.trainable = False
            for layer in model_base.layers[-self.nb_layers_to_learn:]:
                layer.trainable = True

            # ResNet50 model (pre-trained)
            model = model_base.output
            model = tf.keras.layers.AveragePooling2D(pool_size = (7,7))(model)
            model = tf.keras.layers.Flatten()(model)

            # layer FC 1
            model = tf.keras.layers.Dense(units=128, activation="relu", name="Dense_1")(model)
            model =  tf.keras.layers.Dropout(rate=0.5, name="Dropout_Dense_1")(model)

            # layer 2
            model = tf.keras.layers.Dense(units=64, activation="relu", name="Dense_2")(model)
            model = tf.keras.layers.Dropout(rate=0.5, name="Dropout_Dense_2")(model)

            # Softmax
            model = tf.keras.layers.Dense(units=2, activation="softmax", name="Softmax")(model)

            model_fine_tuning = tf.keras.Model(inputs=model_base.input, outputs=model, name="FineTuning")

            model_fine_tuning.compile(optimizer=self._optimizer,loss=self._loss, metrics=[self._metrics])

            print("Model fine tuning created")

            return model_fine_tuning





