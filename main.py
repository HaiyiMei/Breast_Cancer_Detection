from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers, callbacks, regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from alexnet import AlexNet
import os

LEARN_RATE = 1e-2
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
epochs = 128
batch_size = 32
SEED = 11
base_data_dir = './data/'

def my_model_fn(model_name='resnet50'):    
    reg = regularizers.l2(0.01)
    if model_name=='vgg16':
        conv_base = VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape)
        conv_base.trainable = False

    elif model_name=='resnet50':
        conv_base = ResNet50(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)
        conv_base.trainable = False

    elif model_name=='inception_v3':
        conv_base = InceptionV3(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
        conv_base.trainable = False
    
    elif model_name=='alexnet':
        conv_base = AlexNet(input_shape=input_shape, reg=reg)

    else:
        raise Exception("[Model Error]: Select a model that is defined in this code!")

    model = models.Sequential()
    model.add(conv_base)
    if model_name=='inception_v3' or model_name=='resnet50':
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.Flatten())

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

def get_callbacks(path_train_log, path_checkpoint):
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            min_lr=1e-6,
                                            factor=0.5, 
                                            patience=3,
                                            verbose=1,
                                            mode='auto')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=30,
                                            verbose=1,
                                            min_delta=0.001)
    csv_logger = callbacks.CSVLogger(path_train_log)
    checkpointer = callbacks.ModelCheckpoint(filepath=path_checkpoint, 
                                            save_best_only=True,
                                            save_weights_only=True)
    return [reduce_lr, early_stopping, csv_logger, checkpointer]


if __name__ == "__main__":
    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 255,
        data_format=None,
        validation_split=0.0)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    for MODEL in ['resnet50', 'inception_v3', 'vgg16', 'alexnet']:
        for zoom_size in ['40', '100', '200', '400']:
            path_train_log = './log/csv/{}_{}_train_log.csv'.format(MODEL, zoom_size)
            path_checkpoint = './log/checkpoint/{}_{}_checkpoint.h5'.format(MODEL, zoom_size)

            validation_data_dir = os.path.join(base_data_dir, zoom_size, 'test')
            train_data_dir = os.path.join(base_data_dir, zoom_size, 'train')

            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical',
                seed=SEED)

            validation_generator = validation_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode='categorical',
                seed=SEED)

            model = my_model_fn(model_name=MODEL)
            model.fit_generator(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // batch_size,
                callbacks=get_callbacks(path_train_log, path_checkpoint))

            model.load_weights(path_checkpoint)
            scores = model.evaluate_generator(validation_generator)
            print('Test accuracy: ', scores[1])
            with open(path_train_log, 'a') as f:
                f.write('Test accuracy: {}\n'.format(scores[1]))