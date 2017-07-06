import os
import numpy as np

os.environ['KERAS_BACKEND']="theano"
os.environ['THEANO_FLAGS']="floatX=float32,device=gpu"

from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
K.set_image_dim_ordering('th')

class Vgg16:

    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))

    def __init__(self):
        """
        Constructor: creates Vgg16 model and loads weights for later use
        """
        model = self.model = Sequential()
        model.add(Lambda(self.vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(7, activation='softmax'))

        fname = 'vgg_weights_ep8.h5'
        model.load_weights(os.path.abspath(fname))

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def vgg_preprocess(self, x):
        x = x - self.vgg_mean  # subtract mean
        return x[:, ::-1]  # reverse axis bgr->rgb


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical',
                    target_size=(224, 224)):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def get_data(self, path, target_size=(224, 224)):
        batches = self.get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
        return np.concatenate([batches.next() for i in range(batches.samples)])


    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, nb_epoch=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, steps_per_epoch=batches.samples, epochs=nb_epoch,
                validation_data=val_batches, validation_steps=val_batches.samples)

    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.samples)