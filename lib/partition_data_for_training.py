import os
import shutil

from lib.common import get_dataset, split_list

TEMP_AUGMENTED_SPECTS_PATH = "..\\data\\temp_augmented_spects\\"
AUGMENTED_SPECTS_PATH = "..\\data\\augmented_spects\\"
READY_TRAIN_FOLDER = "..\\data\\ready_to_train\\"

folders = ["train", "valid", "sample\\train", "sample\\test"]
category_folder = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

def create_directories(root_path, temp_augm_spec_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for folder in folders:
        for category in category_folder:
            os.makedirs(os.path.join(root_path, folder, category), exist_ok=True)

    if not os.path.exists(temp_augm_spec_path):
        shutil.copytree(AUGMENTED_SPECTS_PATH, temp_augm_spec_path)


def prepare_for_train(ready_train_folder, temp_augm_spec_path):
    spects_tree = get_dataset(temp_augm_spec_path)
    train_percentage = 0.8
    for key in spects_tree:
        train_list, valid_list = split_list(spects_tree[key], measure=train_percentage)
        for spect in train_list:
            source = os.path.join(temp_augm_spec_path, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "train\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)
        for spect in valid_list:
            source = os.path.join(temp_augm_spec_path, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "valid\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)

if __name__ == '__main__':
    create_directories(READY_TRAIN_FOLDER, TEMP_AUGMENTED_SPECTS_PATH)
    prepare_for_train(READY_TRAIN_FOLDER, TEMP_AUGMENTED_SPECTS_PATH)




