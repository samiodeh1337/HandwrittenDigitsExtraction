import numpy as np
import os,sys
from preprocessing import *
import tensorflow as tf 
import keras as ker
from Painter import *
from segmentation import *
cwd = os.getcwd()
sys.path.append(cwd)

print('Validation accuracy: 93.67')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Convolution2D(filters=40,kernel_size=5,strides=(1,1),activation=tf.nn.relu,input_shape=(28,28,1),use_bias=False))
model.add(  tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(  tf.keras.layers.Convolution2D(filters=80,kernel_size=5,strides=(1,1),activation=tf.nn.relu,input_shape=(12,12,40),use_bias=False))
model.add(  tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(  tf.keras.layers.Flatten())
model.add(  tf.keras.layers.Dense(512,activation='relu',use_bias=False))
model.add(  tf.keras.layers.Dense(11, activation='softmax',use_bias=False))

#optimizer = ker.optimizers.SGD(lr=0.01, decay=0.1, momentum=0.1, nesterov=False)
model.compile(optimizer ="SGD",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.load_weights('my_model.h5')

import tkinter as tk
from tkinter import filedialog
Paint()
datapath = cwd + '/image_1.png'

if os.path.isfile(datapath):
    crop_text_image(image_path=datapath,model=model)
try: 
    os.remove(datapath)
except: pass
