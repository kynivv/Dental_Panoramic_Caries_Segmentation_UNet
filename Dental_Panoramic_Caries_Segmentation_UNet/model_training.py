import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2

from tensorflow import keras
from keras import layers
from zipfile import ZipFile
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


# Constants & Hyperparameters
BATCH_SIZE = 5
EPOCHS = 15
IMG_WIDTH = 1536
IMG_HEIGHT = 768
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
SPLIT = 0.25


# Data Extraction
with ZipFile('teeth.zip') as t_scans:
    t_scans.extractall('data')


# Data Preprocessing
X = []
Y = []

images_path = 'data/images_cut'
labels_path = 'data/labels_cut'

images = glob(f'{images_path}/*.png')
for image in images:
    img = cv2.imread(image)

    X.append(img)

labels = glob(f'{labels_path}/*.png')
for label in labels:
    lab = cv2.imread(label)

    Y.append(lab)

X = np.asarray(X)
Y = np.asarray(Y)

X = X.astype('float32')
X = Y.astype('float32')


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    shuffle= True,
                                                    random_state= 24
                                                    )


# U-NET Model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same',),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding= 'same'),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(64, (2, 2), strides= (2, 2), padding= 'same'),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2D(3, (1, 1), activation='sigmoid')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'],
              )


# Callbacks
checkpoint = ModelCheckpoint('output/model.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False,
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          verbose= 1,
          validation_data= (X_test, Y_test),
          callbacks= checkpoint
          )
