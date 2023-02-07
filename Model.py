import sys

sys.path.append("./")

import os
import time
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import datetime
from codecarbon import EmissionsTracker
import csv
from utils import DataLoader as dl
from PIL import Image
# %matplotlib inline


"""
    This class is used to train, evaluate, save, load the model.
    Class variables:
        _model : tf.keras.Model
        _model_name : str
        _optimizer : str
        _loss : tf.keras.losses
        _metrics : str
        _emission : float = Emission product during the training in kgCO2e
        _eval_infos : [true_positive, false_positive, true_negative, false_negative, accuracy]
"""
class Model:
    _model = None
    _model_name : str = "Base Model"
    _emission_train : float = 0
    _eval_infos = None
    _data_loader = dl.DataLoader()


    """
        This function initialize our class
    """
    def __init__(self,model_name,optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(),metrics='accuracy'):
        self._model_name = model_name
        

    """
        This function clean the logs for tensorboard
    """
    def clean_logs(self):
        try:
            os.rmtree('./logs')
        except:
            pass

    """
        This function show the learning rate in a plot
        Function variables:
            history : tf.keras.callbacks.History
            title : str
    """
    def plot_learning_curves(self, history, title):
        print("PLOTING LEARNING CURVES ? ")
        acc = history.history["accuracy"]
        loss = history.history["loss"]
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]
        epochs = range(len(acc))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        fig.suptitle(title, fontsize="x-large")
        ax1.plot(epochs, acc, label="Entraînement")
        ax1.plot(epochs, val_acc, label="Validation")
        ax1.set_title("Accuracy - Données entraînement vs. validation.")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax2.plot(epochs, loss, label="Entraînement")
        ax2.plot(epochs, val_loss, label="Validation")
        ax2.set_title("Perte - Données entraînement vs. validation.")
        ax2.set_ylabel('Perte')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        fig.savefig('./results/plots/'+self._model_name+'.png') 
    

    """
        This function train our model and evaluate the carbon emission of this training. It also plot the learning curves and open tensorboard.
        Function variables:
            epochs : int
            batch_size : int
    """
    def train(self, epochs=100, batch_size=100,patience=10):

        print("Training the model parent")
        self.clean_logs()

        #use dataloader here
        train_gen = self._data_loader.data_retriever("train")
        dev_gen = self._data_loader.data_retriever("dev")

        #tensorboard
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)

        #tracking emissions
        tracker = EmissionsTracker()
        tracker.start()

        #model training
        hist = self._model.fit( 
            train_gen,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=dev_gen,
            verbose=1,
            callbacks=[early_stopping])#,tensorboard_callback])

        self._emission_train  = tracker.stop()
        title = f"{self._model_name} - Learning curves"
        self.plot_learning_curves(hist, title)
        print(f"Emissions: {self._emission_train} kgCO2e")

        #see model in tensorboard
        # tensorboard --logdir ./logs/fit

    """
        This function predict the class of an image.
        Function variables:
            data : data to predict
    """
    def predict(self, data):
        return self._model.predict(data)


    """
        This function save the model in model_name.h5 if there is no path specified and the results of the evaluation are writed in a csv file where all the result of our models are written.
        Function variables:
            path : str
    """
    def save(self, path=None):
        now=datetime.datetime.now()
        now.strftime("%d/%m/%Y %H:%M:%S")
        if path is None:
            path = "./saved_models/{}.h5".format(self._model_name)
        self._model.save(path)
        if not os.path.exists("./saved_models/saved_models_results.csv"):
            with open("./saved_models/saved_models_results.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['model_name','true positive','false positive', 'true negative' , 'false negative' ,'accuracy', 'emission(kgCO2e', 'date'])
        with open("./saved_models/saved_models_results.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self._model_name, self._eval_infos[0],self._eval_infos[1],self._eval_infos[2],self._eval_infos[3], self._eval_infos[4], self._emission, now])


    """
        This function load a model.
        Function variables:
            path : str
    """
    def load(self, path=None):
        if path is None:
            path = "saved_models/{}.h5".format(self._model_name)
        self._model = tf.keras.models.load_model(path)


    """
        This function return the model
    """
    def get_model(self):
        return self._model


    """
        This function return the model name
    """
    def get_model_name(self):
        return self._model_name


    """
        This function set the model name
    """
    def set_model_name(self, model_name):
        self._model_name = model_name


    """
        This function print some useful informations about the model.
    """
    def infos_model(self):
        self._model.summary()
        if self._emission_train != 0:
            print("Emissions during training: {} kgCO2e".format(self._emission_train))
        if self._eval_infos is not None:
            print("Evaluation infos: \nTrue positive: {}\nFalse positive: {}\nTrue negative: {}\nFalse negative: {}\nAccuracy:{}".format(self._eval_infos[0], self._eval_infos[1], self._eval_infos[2], self._eval_infos[3], self._eval_infos[4]))
        # !tensorboard --logdir ./logs/fit


# -------------------- About evaluation --------------------




    """
        This function clean the result folder and create the csv files for the predictions and the evaluation and add the header.
    """
    def clean_result_init(self):
        if os.path.exists('./results/predictions_'+self._model_name+'.csv'):
            os.remove('./results/predictions_'+self._model_name+'.csv')
        with open('./results/predictions_'+self._model_name+'.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path','prediction','ground_truth', 'predicted_output'])
        if not os.path.exists('./results/result_evaluation.csv'):
            with open('./results/result_evaluation.csv', 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['model_name','true positive','false positive', 'true negative' , 'false negative' ,'accuracy', 'emission'])


    """
        Rule : if the probability that there is no fire is lower than LIMIT then we consider that there could be a fire
        Returns True if the rule indicates that there could be a fire
    """
    def no_fire_low(self,no_fire, LIMIT=0.9):
        if no_fire < LIMIT:
            return True
        else:
            return False

    """
        Rule : if the probability that there is a fire is upper than LIMIT then we consider that there could be a fire
        Returns True if the rule indicates that there could be a fire
    """
    def fire_high(self,fire, LIMIT=0.5):
        if fire > LIMIT:
            return True
        else:
            return False

    """
        Rule : count the number of time the rule indicates that there could be a fire 
                2 -> fire for sure
                1 -> maybe fire
                0 -> no fire
        Returns the prediction array according to the rules
    """
    def apply_rules(self, fire, no_fire):
        if self.no_fire_low(no_fire):
            return [1,0]
        else :
            return [0,1]
    
    """
        This function evaluate the model and save the results in 2 csv file : in prediction.csv where the predictions are record and in result_evaluation.csv where the results of the evaluation are record.
        Function variables:
            LIMIT : float : si la probabilité que ce n'est pas un feu est supérieur à LIMIT alors on considère qu'il n'y a pas de feu
    """
    def evaluate(self):
        #tracking emissions
        tracker = EmissionsTracker()
        tracker.start()

        self.clean_result_init()

        predictions = []
        true_negative : int = 0
        false_negative : int = 0
        true_positive : int = 0
        false_positive : int = 0

        # Get the test data
        test_gen = self._data_loader.data_retriever("test")
        # Get the image paths
        image_paths = test_gen.file_paths

        # Keep track of the batch id        
        batch_position = 0
        # For each batch of images
        for image, ground_truth in test_gen:

            # Get the prediction for the batch
            output = self.predict(image)
            print(output)

            # For each image of the batch
            for i in range(0,len(ground_truth)):
             
                real_output = output[i].copy()

                # Apply decision rules
                fire = output[i][0]
                no_fire = output[i][1]
                output[i] = self.apply_rules(fire, no_fire)

                # Count the number of true positive, true negative, false positive and false negative
                if output[i][0] == ground_truth[i][0]:
                    if ground_truth[i][1] == 0:
                        true_positive += 1
                    else:
                        true_negative += 1
                else:
                    if ground_truth[i][1] == 0:
                        false_negative += 1
                    else:
                        false_positive+= 1

                # Add prediction to the tracking list
                predictions.append([image_paths[batch_position+i], output[i], (ground_truth[i]).numpy(), real_output])

            batch_position += len(ground_truth)

        self._emission_test  = tracker.stop() 

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        self._eval_infos = [true_positive, false_positive, true_negative, false_negative, accuracy]
        with open('./results/predictions_'+self._model_name+'.csv', 'a') as f:
            writer = csv.writer(f)
            for prediction in predictions:
                writer.writerow(prediction)
        with open('./results/result_evaluation.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self._model_name, self._eval_infos[0], self._eval_infos[1], self._eval_infos[2], self._eval_infos[3], self._eval_infos[4], self._emission_test])
                

   
    

