import numpy as np
import os, glob, re, shutil
os.environ['KERAS_BACKEND']="theano"
os.environ['THEANO_FLAGS']="floatX=float32,device=cpu"

from keras.utils.np_utils import to_categorical
from vagArchitecture import EmotionRecognition


def put_me_somewhere():
    temp_path = 'data/predict/'
    finetune_model = EmotionRecognition()
    data = finetune_model.get_data(temp_path)
    print('Starting prediction...')
    category_3_folder = ["positive", "neutral", "negative"]

    predictions = finetune_model.model.predict(data, batch_size=4, verbose=1)
    print('Predictions : ', predictions)
    print("=" * 75)
    ## per file
    # for pred in predictions:
    #     print('Prediction : ', pred)
    #     features = np.mean(pred, dtype=np.float32)
    #
    #     print('Features after prediction : ', features)
    #     print('Np argmax after prediction : ', np.argmax(features))
    #     print('Result after prediction : ', category_3_folder[np.argmax(features)])
    #     print("=" * 75)

    ## overall
    features = np.mean(predictions, axis=0, dtype=np.float32)
    print('Features after prediction : ', features)
    print('Np argmax after prediction : ', np.argmax(features))
    print('Result after prediction : ', category_3_folder[np.argmax(features)])
    print("=" * 75)

if __name__ == '__main__':
    put_me_somewhere()
