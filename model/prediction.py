import numpy as np
import os, glob, re, shutil
os.environ['KERAS_BACKEND']="theano"
os.environ['THEANO_FLAGS']="floatX=float32,device=gpu"

from keras.utils.np_utils import to_categorical
from lib.constants.constants import category_3_folder
from lib.preprocess.preprocess_data import extract_audio_from_video, process_spectrograms_dataset, augment_with_overlap_dataset
from network.Architecture import EmotionRecognition
from finetune.Vgg16 import Vgg16



def classification(filename, finetune_model = None):
    temp_path = os.path.join(os.path.abspath(os.path.dirname(filename)), "temp")
    temp_labeled_path = os.path.join(temp_path, "unlabeled")
    if not os.path.exists(temp_path):
        os.makedirs(temp_labeled_path)
    # Extract audio files from video
    extract_audio_from_video(filename, temp_labeled_path, delete = False)
    # Split audio
    augment_with_overlap_dataset(temp_labeled_path, 10, 512, 256)
    # Create spectrograms
    process_spectrograms_dataset(temp_path, 10)

    data = finetune_model.get_data(temp_path)
    print('Starting prediction...')
    features = np.mean(finetune_model.model.predict(data, batch_size=1), axis=0, dtype=np.float32)
    with open("FUCK_IT_test.txt", "a") as myfile:
        text = 'The video %s  audio expresses %s.\n' % (os.path.basename(filename), category_3_folder[np.argmax(features)])
        myfile.write(text)
    shutil.rmtree(temp_path)
    return features


def classification_all(data_path, weights_file, probabilities_filename, labels_filename):
    vgg = Vgg16()
    vgg.model.load_weights(weights_file)

    source = os.path.abspath(data_path)
    dict = {'a': 0, 'd': 0, 'f': 0, 'h': 2, 'n': 1, 'sa': 0, 'su': 2}
    video_no = 0
    vision_probabilities = []
    labels = []

    # parse entire directory and process all .avi files in it
    for file_path in glob.iglob(os.path.join(source, r'**/*.avi'), recursive=True):
        regex = r"([a-zA-Z]+)\d+"
        file_label = re.findall(regex, os.path.split(file_path)[-1])[0]
        for key in dict.keys():
            if key == file_label:
                labels.append(dict[key])
        vision_probabilities.append(classification(file_path, vgg))
        video_no += 1

    vision_probabilities = np.array(vision_probabilities)
    labels = to_categorical(np.array(labels))

    np.save(probabilities_filename, vision_probabilities)
    np.save(labels_filename, labels)

    print(vision_probabilities.shape)
    print(labels.shape)


if __name__ == '__main__':
    data_path = '..\\data\\finetuning_database\\test\\'
    weights = "weights_audio.h5"
    probabilities = 'FUCK_IT_probabilities_test'
    labels = 'FUCK_IT_labels_test'
    classification_all(data_path, weights, probabilities, labels)