                         ###################
                        #                  #
 #######               #  #  #####  #####  #   ###
#       #      #      #   #     #      #   #  #   #
 ###     #    # #    #    #    #      #    #  ####
    #     #  #   #  #     #   #      #     #  #
####       ##     ##      #  #####  #####  #   ###

#############################################
#                   IMPORTS                 #
#############################################

#various
import datetime
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image
import logging

#sklearn
from sklearn.model_selection import train_test_split

#tensorflow
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
#from tensorflow.keras import layers

#keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

#############################################
#                   CONSTANTS               #
#############################################

RSEED = 42

class swizzle_model:

    def __init__(self, 
                 BATCH_SIZE=128, 
                 FRAME_HEIGHT=192,
                 EPOCHS=8,
                 FRAME_WIDTH = 9,
                 INPUTPATH="../data/output/",
                 save_path="../data/model/"):   
        
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.FRAME_WIDTH = FRAME_WIDTH
        self.INPUTPATH = INPUTPATH
        self.save_path = save_path
        self.N_CLASSES = 21
        self.N_STRINGS = 6

        self.load_files()
        
        self.save_folder = self.save_path + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Check for Tensorflow version
        print(tf.__version__)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


    def load_files(self, file_name = str):

        IMAGES = np.load(self.INPUTPATH + 'training_data.npz')
        annots = np.load(self.INPUTPATH + 'training_labels.npz')

        self.logger.info(f"Loading output images and annotations.")
        return IMAGES, annots


    def data_split(self, IMAGES, annots, file_name = str):

        self.IMAGES = IMAGES
        self.annots = annots

        """
        First we have to split our dataset into train and test set. 
        We use 70% for the train set and 30% for the test set.
        Because we need also a validation set we split once more. 
        We take this time 10% of the train set for 
        the validation set and take the rest for training.

        '"""
        self.train_images, self.test_images, self.train_annots, self.test_annots = train_test_split(
            IMAGES['arr_0'], annots['arr_0'], test_size= 0.3, random_state= RSEED )
        self.train_images, self.validate_images,self.train_annots,self.validate_annots = train_test_split(
            self.train_images,self.train_annots, test_size = 0.1,random_state = RSEED)

        self.logger.info(f"Split the data into train, validation and test sets.")


    def catcross_by_string(self,target, output):
        '''The normal categorical_crossentropy was not working with our model, so we had to use a 
        categorical crossentropy which is working string by string.
        It takes:
        It returns: Our loss function'''
        loss = 0
        for i in range(self.N_STRINGS):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss

    
    def softmax_by_string(self,t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.N_STRINGS):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)


    def cnn_swizzle_model(self): 
        #the function of our cnn model
        '''what it takes:
        - a picture with a certain frame height(192pixel) and a frame width(9 pixel)
        - only one color channel, therefore as a grayscale image

        what it deliver:

        An array with the size 6x21. This is representing the 6 different strings of a guitar and 19 different 
        frets of the guitar. The other 2 of the 21 entries represent, if a string is played or not played.

        The different layers we used you can easily extract from below.
        '''      
        swizzle_model = tf.keras.Sequential()
        swizzle_model.add(tf.keras.layers.InputLayer(input_shape=[self.FRAME_HEIGHT, self.FRAME_WIDTH, 1]))
        swizzle_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),activation='relu'))
        swizzle_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        swizzle_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        swizzle_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        swizzle_model.add(tf.keras.layers.Dropout(0.25))   
        swizzle_model.add(tf.keras.layers.Flatten())
        swizzle_model.add(tf.keras.layers.Dense(128, activation='relu'))
        swizzle_model.add(tf.keras.layers.Dropout(0.5))
        swizzle_model.add(tf.keras.layers.Dense(126, activation='relu'))
        swizzle_model.add(tf.keras.layers.Dense(self.N_CLASSES * self.N_STRINGS)) # no activation
        swizzle_model.add(tf.keras.layers.Reshape((self.N_STRINGS, self.N_CLASSES)))
        swizzle_model.add(tf.keras.layers.Activation(self.softmax_by_string))

        self.swizzle_model = swizzle_model
        return swizzle_model


    def compile_model(self):

        '''
        Metric: For our model we will use the accuracy metric, because we want to have o good overall 
        prediction of our model. Besides that, for us every tone has the same importance so all classes
        have the same importance.

        Optimizer: As an optimizer we take the adam optimizer, which is fast enough to handle our data 
        in a short time

        Loss function: For the loss function we used categorical crossentropy because we have multiple classes or labels
        with soft probabilities like [0.5, 0.3, 0.2] and also have a shape like a one-hot-encoded array.
        '''
        metrics =['accuracy']
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss='categorical_crossentropy'
        swizzle_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
   
    
    def train_model(self):

        # Set a learning rate annealer
        '''
        With the ReduceLROnPlateau function from Keras.callbacks, 
        we choose to reduce the Learning Rate by half if the accuracy is not improved after 3 epochs.
        '''
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.0001)

        #for the training we fit our model and use the batch size and epochs from our constants
        history = self.swizzle_model.fit(self.train_images,
                                    self.train_annots,
                                    batch_size=self.BATCH_SIZE,
                                    epochs=self.EPOCHS,
                                    verbose=1,
                                    use_multiprocessing=True,
                                    validation_data=(self.validate_images,self.validate_annots),
                                    callbacks=[learning_rate_reduction],
        )

        score = swizzle_model.evaluate(self.test_images,self.test_annots,verbose=0)
        print('Test Loss : {:.4f}'.format(score[0]))
        print('Test Accuracy : {:.4f}'.format(score[1]))


    def predict_model(self):
        self.model_output = self.swizzle_model.predict(self.test_images)
        self.logger.info(f"swizzle is doing the magic :)")


    def save_model(self):
        # create directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # get path
        if not path:
            path = self.save_path

        # fix path string
        if not path.endswith('/'): 
            path += '/'

        # save files if data is present
            swizzle_model.save(self.save_path + 'swizzle_model')
        self.logger.info(f"swizzle model is being saved :)")


    def save_output(self):     
        # create directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # get path
        if not path:
            path = self.save_path

        # fix path string
        if not path.endswith('/'): 
            path += '/'

        # save files if data is present
        np.save(self.save_path + "model_output.npy", self.model_output, allow_pickle=True, fix_imports=True)
        self.logger.info(f"swizzle model output is being saved :)")