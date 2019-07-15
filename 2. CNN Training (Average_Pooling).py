import sys
import os
import time
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential 
from keras import callbacks

start= time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
    DEV = True
if DEV:
    epochs = 2
else:
    epochs = 30
    
train_data_path = './data/train'
validation_data_path = './data/validation'

"""
Parameters
"""
img_width, img_height = 128,128
batch_size = 30
samples_per_epoch = 240
validation_steps = 60
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 2
classes_num = 3
lr = 0.001

model = Sequential()
model.add(Conv2D(nb_filters1,(conv1_size, conv1_size), padding ="same",
                 input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(pool_size, pool_size)))
model.add(Conv2D(nb_filters2, (conv2_size, conv2_size), padding ="same"))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr, beta_1=0.9,
                                        beta_2=0.999, epsilon=None, decay=0.0,
                                        amsgrad=False),
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/tf-log(epoch=50,lr=0.001,Op=adam)/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/model(epoch=100,lr=0.001,Op=adam)/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('./models/model(epoch=100,lr=0.001,Op=adam)/model.h5')
model.save_weights('./models/model(epoch=100,lr=0.001,Op=adam)/weights.h5')

end = time.time()
dur = end-start
if dur<60:
    print("Execution Time: ", dur," Seconds")
elif dur>0 and dur<3600:
    dur=dur/60
    print("Execution Time: ", dur," Minutes")
else:
    dur=dur/(60*60)
    print("Execution Time: ", dur," Hours")
    
