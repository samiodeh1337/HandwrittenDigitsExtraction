import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd 
import sys ,os
import keras as ker


cwd = os.getcwd()
sys.path.append(cwd)

import pickle
datapath = cwd + '/Manage_Data/english/'

train_both_letters = pd.read_csv(datapath + "emnist-letters-train.csv",header=None)
train_both_digits = pd.read_csv(datapath + "emnist-digits-train.csv",header=None)

test_both_letters = pd.read_csv(datapath + "emnist-letters-test.csv",header=None)
test_both_digits = pd.read_csv(datapath + "emnist-digits-test.csv",header=None)

train_both_letters = train_both_letters.values.astype('float32')
train_both_digits = train_both_digits.values.astype('float32')
test_both_letters = test_both_letters.values.astype('float32')
test_both_digits = test_both_digits.values.astype('float32')


train_letters_images = np.array(train_both_letters[:,1:785])
train_letters_labels = np.full(88800,10,dtype=np.uint8)
train_digits_images = np.array(train_both_digits[:,1:785])
train_digits_labels = np.array(train_both_digits[:,0],dtype=np.uint8)
#####################################################################
test_letters_images = np.array(test_both_letters[:,1:785])
test_letters_labels = np.full(14800,10,dtype=np.uint8)
test_digits_images = np.array(test_both_digits[:,1:785])
test_digits_labels = np.array(test_both_digits[:,0],dtype=np.uint8) #40000

#####################################################################

train_letters_images = train_letters_images.reshape(88800,28,28)
train_digits_images = train_digits_images.reshape(240000,28,28)

test_letters_images = test_letters_images.reshape(14800,28,28)
test_digits_images = test_digits_images.reshape(40000,28,28)



images_all=[]
images_all=np.append(images_all,train_digits_images)
images_all=np.append(images_all,train_letters_images)
images_all=images_all.reshape(240000+88800,28,28)

labels_all=[]
labels_all=np.append(labels_all,train_digits_labels)
labels_all=np.append(labels_all,train_letters_labels)

#shuffle all
shuffler = np.random.permutation(328800)
images_all = images_all[shuffler]
labels_all = labels_all[shuffler]


images_all_tests = []
images_all_tests = np.append(images_all_tests,test_digits_images)
images_all_tests = np.append(images_all_tests,test_letters_images)
images_all_tests = images_all_tests.reshape(14800+40000,28,28)

labels_all_tests =[]
labels_all_tests = np.append(labels_all_tests,test_digits_labels)
labels_all_tests = np.append(labels_all_tests,test_letters_labels)
print(images_all_tests.shape,labels_all_tests.shape)

shuffler = np.random.permutation(14800+40000)
images_all_tests = images_all_tests[shuffler]
labels_all_tests = labels_all_tests[shuffler]

#####################################

#images_all = tf.keras.utils.normalize(images_all, axis=1)
images_all = images_all.reshape(328800,28,28,1)
images_all -= int(np.mean(images_all))
images_all /= int(np.std(images_all))

images_all_tests = images_all_tests.reshape(54800,28,28,1)
images_all_tests -= int(np.mean(images_all_tests))
images_all_tests /= int(np.std(images_all_tests))

#cv2.imshow('ImageWindow', images_all_tests[16])
#cv2.waitKey()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Convolution2D(filters=40,kernel_size=5,strides=(1,1),activation=tf.nn.relu,input_shape=(28,28,1),use_bias=False))
model.add(  tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(  tf.keras.layers.Convolution2D(filters=80,kernel_size=5,strides=(1,1),activation=tf.nn.relu,input_shape=(12,12,40),use_bias=False))
model.add(  tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(  tf.keras.layers.Flatten())
model.add(  tf.keras.layers.Dense(512,activation='relu',use_bias=False))
model.add(  tf.keras.layers.Dense(11, activation='softmax',use_bias=False))

#optimizer = ker.optimizers.SGD(lr=0.01, decay=0.1, momentum=0.1, nesterov=False)
model.compile(optimizer="SGD",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()
model.fit(x=images_all,y=labels_all,epochs =30,shuffle=True,batch_size=150)
#steps_per_epoch=328800
model.save_weights('my_model.h5')

scores = model.evaluate(images_all_tests, labels_all_tests, verbose=0)
print("Validation accuracy: %.2f%%" % (scores[1]*100))


#150 6.27   SGD 10.88
#32 7.87    SGD 7.72
#300 6.89   SGD 11.26
#100 6.68   SGD 8.85
