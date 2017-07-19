import os, json

os.environ['KERAS_BACKEND'] = "theano"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu,force_device=True," \
                             "exception_verbosity=high,nvcc.fastmath=True," \
                             "print_active_device=True"
import numpy as np

np.set_printoptions(precision=4, linewidth=100)

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

K.set_image_dim_ordering('th')

######################################################################
MODEL1_OR_MODEL2 = 0  # set 0 for train Model 1, or 1 for Model 2
EPOCHS = 50
PATH = "\\data"
BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)
######################################################################

class EmotionRecognition:
    def __init__(self, conv_model=MODEL1_OR_MODEL2):
        self.model = None
        self.create(conv_model)

    # Vgg based
    def ConvBlockModel1(self):
        self.model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(256, (7, 7), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(512, (7, 7), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Model 2
    def ConvBlockModel2(self):
        self.model.add(Convolution2D(64, 2, activation='elu', input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(128, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(256, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(512, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))


    def ConvBlock(self):
        self.model.add(Convolution2D(120, (11, 11), strides=(4, 4), activation='relu',
                                     input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        self.model.add(Convolution2D(256, (5, 5), activation='relu', strides=(1, 1)))
        self.model.add(Convolution2D(384, (3, 3), activation='relu', strides=(1, 1)))

    def FCBlockModel1(self):
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))

    def FCBlockModel2(self):
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))


    def create(self, conv_model):
        self.model = Sequential()
        if conv_model == 0:
            self.ConvBlockModel1()
        else:
            self.ConvBlockModel2()
        self.FCBlock()

    def vgg_preprocess(self, x):
        vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        x = x - vgg_mean
        # reverse axis rgb->bgr
        return x[:, ::-1]

    def get_train_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8,
                          class_mode='categorical', target_size=IMAGE_SIZE):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def get_valid_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8,
                          class_mode='categorical', target_size=IMAGE_SIZE):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def get_data(self, path, target_size=IMAGE_SIZE):
        batches = self.get_train_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
        return np.concatenate([batches.next() for i in range(batches.samples)])

    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batches, val_batches, nr_epoch=EPOCHS, callbacks=None):
        self.model.fit_generator(batches, steps_per_epoch=batches.samples, epochs=nr_epoch,
                                 validation_data=val_batches, validation_steps=val_batches.samples, callbacks=callbacks)

    def test(self, path, batch_size=8):
        test_batches = self.get_train_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.samples)


if __name__ == '__main__':
    model = EmotionRecognition()
    batch_size = BATCH_SIZE
    batches = model.get_train_batches(os.path.join(PATH, 'train'), batch_size=batch_size)
    val_batches = model.get_valid_batches(os.path.join(PATH, 'valid'), batch_size=batch_size)
    model.compile()

    # Checkpoint file -> save weights if improvement in validation accuracy
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train
    model.fit(batches, val_batches, nr_epoch=EPOCHS, callbacks=callbacks_list)
