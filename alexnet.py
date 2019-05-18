from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers, callbacks, regularizers

def AlexNet(input_shape, reg):
    model = models.Sequential()
    model.add(layers.Conv2D(96,(11,11), strides=(4,4),
                            input_shape=input_shape, 
                            padding='valid', 
                            activation='relu', 
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(256,(5,5), strides=(1,1),
                            padding='same', 
                            activation='relu', 
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(384,(3,3), strides=(1,1),
                            padding='same', 
                            activation='relu', 
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.Conv2D(384,(3,3), strides=(1,1),
                            padding='same', 
                            activation='relu', 
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.Conv2D(256,(3,3), strides=(1,1),
                            padding='same', 
                            activation='relu', 
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    return model
