import sys

sys.path.append("./")

import os
import time
import datetime

import pandas as pd
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
    _emission : float = 0
    _eval_infos = None
    _data_loader = dl.DataLoader()


    """
        This function initialize our class
    """
    def __init__(self,model_name,optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(),metrics='accuracy'):
        self._model_name = model_name
        

#usefull ?
    # def _build_model(self,load=False):
    #     print( "Building model parent")
    #     if load:
    #         return self.load()
    #     else:
    #         raise NotImplementedError

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

        self._emission  = tracker.stop()
        title = f"{self._model_name} - Learning curves"
        self.plot_learning_curves(hist, title)
        print(f"Emissions: {self._emission} kgCO2e")

        #see model in tensorboard
        # tensorboard --logdir ./logs/fit


    """
        This function clean the result folder and create the csv files for the predictions and the evaluation and add the header.
    """
    def clean_result(self):
        if not os.path.exists('./results'):
            os.makedirs('./results')
            try:
                with open('./result/predictions.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_name','prediction','ground_truth'])
            except:
                print("Error with predictions file")
            try:
                with open('./results/result_evaluation.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['model_name','true positive','false positive', 'true negative' , 'false negative' ,'accuracy', 'emission'])
            except:
                print("Error with result_evaluation file")

    """
        This function evaluate the model and save the results in 2 csv file : in prediction.csv where the predictions are record and in result_evaluation.csv where the results of the evaluation are record.
        Function variables:
            LIMIT : float : si la probabilité que ce n'est pas un feu est supérieur à LIMIT alors on considère qu'il n'y a pas de feu
    """
    def evaluate(self,LIMIT=0.02):
        self.clean_result()
        predictions = []
        true_negative : int = 0
        false_negative : int = 0
        true_positive : int = 0
        false_positive : int = 0
        # Get the test data
        test_gen = self._data_loader.data_retriever("test")
        #For each test image        
        image_id = 0
        for image, ground_truth in test_gen:
            # Get the prediction
            output = self.predict(image)
            for i in range(len(ground_truth)):
                predictions.append([image_id, output[i], (ground_truth[i]).numpy()])
                image_id += 1
                # Si la proba qu'il n'y ait pas de feu est assez haute, on est assuré qu'il n'y en a pas
                if output[i][0]>LIMIT:
                    output[i]=[1,0]
                else:
                    output[i]=[0,1]
                # On compare la prédiction avec le ground truth pour incrémenter la matrice de confusion
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
                
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        self._eval_infos = [true_positive, false_positive, true_negative, false_negative, accuracy]
        if not os.path.exists('./results'):
            os.makedirs('./results')
        with open('./results/predictions.csv', 'a') as f:
            writer = csv.writer(f)
            for prediction in predictions:
                writer.writerow(prediction)
        with open('./results/result_evaluation.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self._model_name, self._eval_infos[0], self._eval_infos[1], self._eval_infos[2], self._eval_infos[3], self._eval_infos[4], self._emission])
                

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
        if self._emission != 0:
            print("Emissions during training: {} kgCO2e".format(self._emission))
        if self._eval_infos is not None:
            print("Evaluation infos: \nTrue positive: {}\nFalse positive: {}\nTrue negative: {}\nFalse negative: {}\nAccuracy:{}".format(self._eval_infos[0], self._eval_infos[1], self._eval_infos[2], self._eval_infos[3], self._eval_infos[4]))
        # !tensorboard --logdir ./logs/fit

    

# """
#     Main function
# """
# if __name__ == "__main__":
#     model = Model("base_model")
#     model.train("dataset/data/annotations/train.csv", "data/annotations/dev.csv")
#     model.infos_model()
#     model.evaluate("dataset/data/annotations/test.csv")
#     model.save()


    

