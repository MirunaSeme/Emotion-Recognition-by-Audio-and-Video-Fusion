import os
import shutil

from lib.common import get_dataset, split_list

TEMP_AUGMENTED_SPECTS_PATH = "..\\data\\augmented\\cropped_augmented_with_overlap_spects\\"
AUGMENTED_SPECTS_PATH = "..\\data\\augmented_spects\\"
READY_TRAIN_FOLDER = "..\\data\\augmented\\ready_to_train_cropped_augmented_with_overlaps\\"

folders = ["train", "valid", "sample\\train", "sample\\test", "test"]
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
    for key in spects_tree:
        temp_list, test_list = split_list(spects_tree[key], measure=0.8)
        train_list, valid_list = split_list(temp_list, measure=0.52)
        for spect in train_list:
            source = os.path.join(temp_augm_spec_path, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "train\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)
        for spect in valid_list:
            source = os.path.join(temp_augm_spec_path, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "valid\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)
        for spect in test_list:
            source = os.path.join(temp_augm_spec_path, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "test\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)

if __name__ == '__main__':
    create_directories(READY_TRAIN_FOLDER, TEMP_AUGMENTED_SPECTS_PATH)
    # prepare_for_train(READY_TRAIN_FOLDER, TEMP_AUGMENTED_SPECTS_PATH)




