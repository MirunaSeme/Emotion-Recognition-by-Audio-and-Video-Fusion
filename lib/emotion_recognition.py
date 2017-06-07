import os, json
os.environ['KERAS_BACKEND']="theano"
os.environ['THEANO_FLAGS']="floatX=float32,device=gpu,force_device=True,lib.cnmem=0.8"
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

class EmotionRecognition():

	def __init__(self):
		self.create()

	def ConvBlock(self):
		model.add(Convolution2D(120, 11, 11, activation='relu'))
		model.add(MaxPooling2D((3,3), strides=(2,2)))

		model.add(Convolution2D(256, 5, 5, activation='relu'))
		model.add(Convolution2D(384, 3, 3, activation='relu'))

	def FCBlock(self)
		model.add(Dense(2048, activation='relu'))
		model.add(Dense(2048, activation='relu'))
		model.add(Dense(7, activation='softmax'))

	def create(self):
		self.model = Sequential()
		self.ConvBlock()
		self.FCBlock()



