import os
import shutil

from lib.common.common import get_dataset, split_list

def prepare_for_train(ready_train_folder, spectrogram_folder, subtree):
    spects_tree = get_dataset(spectrogram_folder, subtree)
    for key in spects_tree:
        temp_list, test_list = split_list(spects_tree[key], measure=0.8)
        train_list, valid_list = split_list(temp_list, measure=0.52)
        for spect in train_list:
            source = os.path.join(spectrogram_folder, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "train\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)
        for spect in valid_list:
            source = os.path.join(spectrogram_folder, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "valid\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)
        for spect in test_list:
            source = os.path.join(spectrogram_folder, os.path.basename(key), os.path.basename(spect))
            destination = os.path.join(ready_train_folder, "test\\", os.path.basename(key), os.path.basename(spect))
            shutil.copy(source, destination)







