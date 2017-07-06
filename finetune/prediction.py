import numpy as np
import os, glob, re, shutil
os.environ['KERAS_BACKEND']="theano"
os.environ['THEANO_FLAGS']="floatX=float32,device=gpu"

from keras.utils.np_utils import to_categorical

from lib.constants.constants import category_3_folder
from lib.preprocess.preprocess_data import extract_audio_from_video, process_spectrograms_dataset, augment_with_overlap_dataset
from finetune.Vgg16 import Vgg16
from network.Architecture import EmotionRecognition



def classification(filename, finetune_model = None):
    temp_path = os.path.join(os.path.abspath(os.path.dirname(filename)), "temp")
    temp_labeled_path = os.path.join(temp_path, "unlabeled")
    if not os.path.exists(temp_path):
        os.makedirs(temp_labeled_path)
    # Extract audio files from video
    extract_audio_from_video(filename, temp_labeled_path, delete = False)
    # Split audio
    # augment_with_overlap_dataset(temp_labeled_path, 10, 512, 256)
    # Create spectrograms
    process_spectrograms_dataset(temp_path, 10)

    batches, preds = finetune_model.test(temp_path, batch_size=16)
    print("[TEST", preds)
    # print("Model :", finetune_model)
    # data = finetune_model.get_data(temp_path)
    # print("Data: ", data)
    # print('Starting prediction...')
    # features = finetune_model.model.model.predict(data, batch_size=32, verbose=1)
    # print("[DONE] The array of prediction: ", features)
    # with open("prediction_file_new_model.txt", "a") as myfile:
    #     text = 'The video %s  audio expresses %s.\n' % (os.path.basename(filename), category_3_folder[np.argmax(features)])
    #     myfile.write(text)
    # shutil.rmtree(temp_path)
    # return features


def classification_all():
    vgg = EmotionRecognition()
    vgg.model.load_weights("vgg_weights_ep8.h5")

    source = os.path.abspath('..\\data\\finetuning_database\\test\\')
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

    np.save('audio_probabilities_train_new_model', vision_probabilities)
    np.save('labels_audio_train_new_model', labels)

    print(vision_probabilities.shape)
    print(labels.shape)


if __name__ == '__main__':

    classification_all()