import os
import tensorflow as tf
import csv
import shutil


class DataLoader :
    DATADIR : str
    ANNOTATIONSDIR : str
    TRAINDIR : str
    TESTDIR : str
    DEVDIR : str

    def __init__(self):
        self.DATADIR  = "./dataset/data_preprocessed"
        self.ANNOTATIONSDIR = os.path.join(self.DATADIR, "annotations")
        self.TRAINDIR = os.path.join(self.DATADIR, "images/train")
        self.TESTDIR  = os.path.join(self.DATADIR, "images/test")
        # self.DEVDIR = os.path.join(self.DATADIR, "images/dev")

    """
        returns 1 if the image is a fire, 0 if not
    """
    def fire_or_not(self, img_name, annotations_path):
        with open(annotations_path) as csvfile:
            for line in csvfile:
                if img_name in line:
                    return int(line[-2])
                
    """
        move images in the right folder according to the annotation
    """
    def move_directory(self, images_path, annotations_path):
        # remove the class folder if it exists
        if os.path.exists(os.path.join(images_path,"class")):
            shutil.rmtree(os.path.join(images_path,"class"), ignore_errors=False, onerror=None)
        # create the class folder
        images = os.listdir(images_path)
        # Create the Fire and NotFire folders
        os.makedirs(os.path.join(images_path,"class/Fire"))
        os.makedirs(os.path.join(images_path,"class/NotFire"))
        # For each image, move it in the right folder
        for i in images:
            origin_path = os.path.join(images_path, i)
            if self.fire_or_not(i, annotations_path) == 1:
                fire_path = os.path.join(images_path,"class/Fire")
                shutil.copy(origin_path, fire_path)
            else:
                not_fire_path = os.path.join(images_path,"class/NotFire")
                shutil.copy(origin_path,not_fire_path)


    """
        generate a dataset from the images in the images_path directory
    """
    def generate_dataset(self, images_path,subset, validation_split):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            images_path, 
            labels="inferred", 
            label_mode='categorical',
            class_names=["fire","no_fire"],
            color_mode='rgb', 
            batch_size=32, 
            image_size=(224, 224), 
            shuffle=True, 
            seed=42, 
            validation_split=validation_split , 
            subset=subset, 
            interpolation='bilinear', 
            follow_links=False, 
            crop_to_aspect_ratio=False
        )
        print("Dataset generated from directory : ",images_path,"...")
        #shutil.rmtree(os.path.join(images_path,"class"), ignore_errors=False, onerror=None)
        return dataset

    def data_retriever(self,dir):
        data_path = ""
        anno_path = ""
        if dir == "train" : 
            data_path = self.TRAINDIR
            print("Start to retrieve data from directory : ",data_path,"...")
            return self.generate_dataset(data_path, "training", 0.2)
        elif dir == "test" :
            data_path = self.TESTDIR
            print("Start to retrieve data from directory : ",data_path,"...")
            return self.generate_dataset(data_path,None, None)
        elif dir == "dev" :
            data_path = self.TESTDIR
            print("Start to retrieve data from directory : ",data_path,"...")
            return self.generate_dataset(data_path, "training", 0.2)
        else :
            print("Error: wrong dir")
            raise ValueError