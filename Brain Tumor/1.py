import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
os.listdir('C:/Users/hrith/PycharmProjects/brain tumor/Data_set')
im =Image.open('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/no/1 no.jpeg').resize((128,128))
im
im =Image.open('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/yes/Y1.jpg').resize((128,128))
im
yes=os.listdir('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/yes')
no=os.listdir('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/no')
data=np.concatenate([yes,no])
len(data)==len(yes)+len(no)
target_x=np.full(len(yes),1)
target_y=np.full(len(no),0)
data_target=np.concatenate([target_x,target_y])
len(data_target)==len(target_x)+len(target_y)
len(data_target)==len(data)
data_target
data
yes_values=os.listdir('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/yes')
no_values=os.listdir('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/no')
X_data =[]
for file in yes_values:
    #face = misc.imread('../input/brain_tumor_dataset/yes/'+file)
    img = cv2.imread('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/yes/'+file)
    face = cv2.resize(img, (32, 32) )
    (b, g, r)=cv2.split(face)
    img=cv2.merge([r,g,b])
    X_data.append(img)
#X_data =[]
for file in no_values:
    #face = misc.imread('../input/brain_tumor_dataset/yes/'+file)
    img = cv2.imread('C:/Users/hrith/PycharmProjects/brain tumor/Data_set/no/'+file)
    face = cv2.resize(img, (32, 32) )
    (b, g, r)=cv2.split(face)
    img=cv2.merge([r,g,b])
    X_data.append(img)
len(X_data)==len(data)==len(data_target)
X = np.squeeze(X_data)
X.shape
# normalize data
X = X.astype('float32')
X /= 255
data_target
(x_train, y_train), (x_test, y_test) = (X[:190],data_target[:190]) , (X[190:] , data_target[190:])
(x_valid , y_valid) = (x_test[:63], y_test[:63])
#(x_test, y_test) = (x_test[63:], y_test[63:])
model = tf.keras.Sequential()
# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=9, padding='same', activation='relu', input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.45))
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=9,padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=9, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Take a look at the model summary
model.summary()
model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(),
             metrics=['acc'])
model.fit(x_train,
         y_train,
         batch_size=128,
         epochs=150,
         validation_data=(x_valid, y_valid),)
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model,show_shapes = True).create(prog='dot', format='svg'))
model.save('brain_tumor.hdf5')
model.save_weights('myweights_Brain_tumor.h5')



