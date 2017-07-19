import os, json

os.environ['KERAS_BACKEND'] = "theano"
os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu,force_device=True," \
                             "exception_verbosity=high," \
                             "print_active_device=True"
# extra THEANO_FLAGS = lib.cnmem=0.9,optimizer=fast_compile
# openmp=True,gpuarray.sched=multi

from glob import glob
import numpy as np

np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

######################################################################
VGG_OR_PIU_OR_TUTU = 2  # set 0 for VGG, 1 for PIU, 2 for TUTU
EPOCHS = 15
# PATH = "..\\data\\3classes\\Tutu_with_pitch"
PATH = "data/"
BATCH_SIZE = 4
# IMAGE_SIZE = (16, 16)
IMAGE_SIZE = (32, 32)
# IMAGE_SIZE = (64, 64)
# IMAGE_SIZE = (128, 128)
# IMAGE_SIZE = (256, 256)
######################################################################

class EmotionRecognition:
    def __init__(self, conv_model=VGG_OR_PIU_OR_TUTU):
        self.model = None
        self.create(conv_model)

    def ConvBlockTutu(self):
        self.model.add(Convolution2D(64, 2, activation='elu', input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(128, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(256, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))

        self.model.add(Convolution2D(512, 2, activation='elu'))
        self.model.add(MaxPooling2D(2))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))


    def ConvBlockVGGBased(self):
        self.model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Convolution2D(128, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #
        # self.model.add(ZeroPadding2D((1, 1)))
        # self.model.add(Convolution2D(256, (7, 7), activation='relu'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #
        # self.model.add(ZeroPadding2D((1, 1)))
        # self.model.add(Convolution2D(512, (7, 7), activation='relu'))
        # self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def ConvBlock(self):
        # self.model.add(Lambda(self.vgg_preprocess, input_shape=(3, 256, 256), output_shape=(3, 256, 256)))
        self.model.add(Convolution2D(120, (11, 11), strides=(4, 4), activation='relu',
                                     input_shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])))
        self.model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        self.model.add(Convolution2D(256, (5, 5), activation='relu', strides=(1, 1)))
        self.model.add(Convolution2D(384, (3, 3), activation='relu', strides=(1, 1)))

    def FCBlock(self):
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        # self.model.add(Dense(3, activation='softmax', input_shape=(3, 256, 256)))
        self.model.add(Dense(3, activation='softmax'))

    def create(self, conv_model):
        self.model = Sequential()
        if conv_model == 0:
            self.ConvBlockVGGBased()
            self.FCBlock()
        elif conv_model == 1:
            self.ConvBlock()
            self.FCBlock()
        else:
            self.ConvBlockTutu()


    def vgg_preprocess(self, x):
        vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
        x = x - vgg_mean
        return x[:, ::-1]  # reverse axis rgb->bgr

    def get_train_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8,
                          class_mode='categorical',
                          target_size=IMAGE_SIZE):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def get_valid_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8,
                          class_mode='categorical',
                          target_size=IMAGE_SIZE):
        return gen.flow_from_directory(path, target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def get_data(self, path, target_size=IMAGE_SIZE):
        batches = self.get_train_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
        # print("Batches: ", [batches.next() for i in range(batches.samples)])
        return np.concatenate([batches.next() for i in range(batches.samples)])

    def compile(self, lr=0.001):
        if VGG_OR_PIU_OR_TUTU == 2:
            print("Compiling for Tutu ...")
            self.model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
            return
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
    print("=" * 75)
    print("=== Model Summary")
    model.model.summary()
    print("=" * 75)
    print("Compiling finished")
    print("Started fitting ...")
    # model.fit(batches, val_batches)
    # model.model.save(PATH + "/model.container", overwrite=True, include_optimizer=True)
    # model.model.save_weights(PATH + "/model.weights", overwrite=True)
    # print("=" * 75, "[DONE]")

    model_vgg_or_piu = "vgg"
    if VGG_OR_PIU_OR_TUTU == 1:
        model_vgg_or_piu = 'piu'
    elif VGG_OR_PIU_OR_TUTU == 2:
        model_vgg_or_piu = 'tutu'

    # for epoch in range(EPOCHS):
    #     print("=" * 75)
    #     print("Running epoch: %d" % epoch)
    #     model.fit(batches, val_batches, nr_epoch=1)
    #     latest_weights_filename = '%s_weights_ep%d.h5' % (model_vgg_or_piu, epoch)
    #     latest_container_filename = "%s_container_ep%d" % (model_vgg_or_piu, epoch)
    #     model.model.save_weights(os.path.join(PATH, latest_weights_filename))
    #     model.model.save_weights(os.path.join(PATH, latest_container_filename))
    #     print("Completed %s fit operations" % epoch)


    print("=" * 75)

    # checkpoint
    filepath = "weights-improvement-%s-{epoch:02d}-{val_acc:.2f}.hdf5" % model_vgg_or_piu
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(batches, val_batches, nr_epoch=EPOCHS, callbacks=callbacks_list)

    print("=" * 75, "[DONE]")
