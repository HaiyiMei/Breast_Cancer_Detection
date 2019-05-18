import matplotlib
import os
matplotlib.use('TkAgg')
from matplotlib import pyplot
pyplot.rcParams['font.sans-serif']=['SimHei']
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.python.keras.layers import (
    GlobalAveragePooling2D, AveragePooling2D
)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import Callback

IMG_WIDTH, IMG_HEIGHT = 224, 224
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
LEARN_RATE = 3e-4
SEED = 11
dropout = 0.5
reg = l2(0.01)

def create_model():
    conv_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE)
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu',
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu',
                            kernel_initializer='uniform',
                            kernel_regularizer=reg))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax',
                           kernel_initializer='uniform',
                           kernel_regularizer=reg))
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=LEARN_RATE, momentum=0.9),
                metrics=['accuracy'])
    model.summary()
    return model

def plot_predictions(images, predictions, true_labels):
    n = images.shape[0]
    nc = int(np.ceil(n / 4))
    fig = pyplot.figure(figsize=(4,3))
    f, axes = pyplot.subplots(nc, 4)
    f.tight_layout()
    for i in range(nc * 4):
        y = i // 4
        x = i % 4
        axes[x, y].axis('off')

        confidence = np.max(predictions[i])
        conf = confidence if confidence>0.5 else 1-confidence
        if i > n:
            continue
        axes[x, y].imshow(images[i])
        axes[x, y].set_title("预测:{}(实际:{})\n 概率:{:.3f}".format(
            LABEL_NAMES[np.argmax(predictions[i])],
            LABEL_NAMES[np.argmax(true_labels[i])],
            conf
        ), color=("green" if np.argmax(predictions[i]) == np.argmax(true_labels[i]) else "red"))
    pyplot.gcf().set_size_inches(8, 8)  
    pyplot.savefig("predict.png", dpi=1000)

LABEL_NAMES = ['良性', '恶性']
MODEL =  'resnet50'
zoom_size = '100'

base_data_dir = './data/'
val_path = os.path.join(base_data_dir, zoom_size, 'test')
path_checkpoint = './log/checkpoint/{}_{}_checkpoint.h5'.format(MODEL, zoom_size)

model = create_model()
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(val_path,
                                                batch_size=16,
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                seed=SEED,
                                                class_mode='categorical')
model.load_weights(path_checkpoint)

x_test, y_test = val_generator.next()
plot_predictions(
    x_test, 
    model.predict(x_test),
    y_test.astype(np.int)
)