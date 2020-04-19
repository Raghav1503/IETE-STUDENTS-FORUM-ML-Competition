import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

train_dir = r'C:\Users\ragha\Desktop\New folder\ietemlcompetition\Dataset\train'
valid_dir = r'C:\Users\ragha\Desktop\New folder\ietemlcompetition\Dataset\valid'

num_classes = len(os.listdir(train_dir))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(64, 64),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    color_mode='rgb')

res_base = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(64, 64, 3))

model = models.Sequential()
model.add(res_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=20)

model.save('test1.h5')






