import shutil, re
from os.path import basename, splitext, dirname, join, exists
from os import mkdir, listdir

from lib.common.common import get_dataset, split_list, create_training_directories
from lib.constants.constants import category_7_folder

VIDEO_DATABASE = "..\\..\\data\\video_database"
TRAIN_VIDEO_NUMBER = 10
VALID_VIDEO_NUMBER = 2
TEST_VIDEO_NUMBER = 3

def partition_train_strict_percentage(ready_train_folder, spectrogram_folder, subtree):
    spects_tree = get_dataset(spectrogram_folder, subtree)
    for key in spects_tree:
        temp_list, test_list = split_list(spects_tree[key], measure=0.8)
        train_list, valid_list = split_list(temp_list, measure=0.52)
        for spect in train_list:
            source = join(spectrogram_folder, basename(key), basename(spect))
            destination = join(ready_train_folder, "train\\", basename(key), basename(spect))
            shutil.copy(source, destination)
        for spect in valid_list:
            source = join(spectrogram_folder, basename(key), basename(spect))
            destination = join(ready_train_folder, "valid\\", basename(key), basename(spect))
            shutil.copy(source, destination)
        for spect in test_list:
            source = join(spectrogram_folder, basename(key), basename(spect))
            destination = join(ready_train_folder, "test\\", basename(key), basename(spect))
            shutil.copy(source, destination)

def partition_train_fusion_synchronization(root_path):
    # root_path = should be the place with new program: constant_dir["root"]
    if not exists(root_path):
        mkdir(root_path)
        create_training_directories(root_path, category_7_folder)

    video_database = get_dataset(VIDEO_DATABASE, 4)
    for index, key in enumerate(video_database):
        for videofile in video_database[key]:
            video_name_list = re.split('(\d+)', basename(videofile))
            if video_name_list[0] == "a":
                move_file(videofile, root_path, "anger\\", int(video_name_list[1]), "a", index)
            if video_name_list[0] == "d":
                move_file(videofile, root_path, "disgust\\", int(video_name_list[1]), "d", index)
            if video_name_list[0] == "f":
                move_file(videofile, root_path, "fear\\", int(video_name_list[1]), "f", index)
            if video_name_list[0] == "su":
                move_file(videofile, root_path, "surprise\\", int(video_name_list[1]), "su", index)
            if video_name_list[0] == "sa":
                move_file(videofile, root_path, "sadness\\", int(video_name_list[1]), "sa", index)
            if video_name_list[0] == "h":
                move_file(videofile, root_path, "happiness\\", int(video_name_list[1]), "h", index)
            if video_name_list[0] == "n":
                move_file(videofile, root_path, "neutral\\", int(video_name_list[1]), "n", index, double=True)

def group_positive(root_path):
    # Happiness & Surprise
    happiness_folder = join(root_path, "happiness\\")
    surprise_folder = join(root_path, "surprise\\")
    positive_folder = join(root_path, "positive\\")
    if not exists(positive_folder):
        mkdir(positive_folder)

        # Process happiness
        happiness_tree = get_dataset(happiness_folder, 7)
        for key in happiness_tree:
            for index, audio in enumerate(happiness_tree[key]):
                replaced_audio_name = basename(splitext(audio)[0]).replace('h', 'p') + ".avi"
                new_audio_path = join(dirname(positive_folder), replaced_audio_name)
                shutil.copy(audio, new_audio_path)
                # print(new_audio_path)
            new_positive_index = index + 2

        # Process surprise
        surprise_tree = get_dataset(surprise_folder, 7)
        for key in surprise_tree:
             for index, audio in enumerate(surprise_tree[key]):
                 replaced_audio_name = "p" + str(new_positive_index + index) + ".avi"
                 new_audio_path = join(dirname(positive_folder), replaced_audio_name)
                 shutil.move(audio, new_audio_path)
                 # print(new_audio_path)

        shutil.rmtree(happiness_folder)
        shutil.rmtree(surprise_folder)


def group_negative(root_path):
    # Sadness & Fear & Anger
    sadness_folder = join(root_path, "sadness\\")
    fear_folder = join(root_path, "fear\\")
    anger_folder = join(root_path, "anger\\")
    disgust_folder = join(root_path, "disgust\\")
    negative_folder = join(root_path, "negative\\")

    if not exists(negative_folder):
        mkdir(negative_folder)

        # Process sadness
        sadness_tree = get_dataset(sadness_folder, 7)
        for key in sadness_tree:
            for index, audio in enumerate(sadness_tree[key]):
                replaced_audio_name = basename(splitext(audio)[0]).replace('sa', 'neg') + ".avi"
                new_audio_path = join(dirname(negative_folder), replaced_audio_name)
                shutil.copy(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = index + 2

        # Process fear
        fear_tree = get_dataset(fear_folder, 7)
        for key in fear_tree:
            for index, audio in enumerate(fear_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".avi"
                new_audio_path = join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = new_negative_index + index + 1

        # Process anger
        anger_tree = get_dataset(anger_folder, 7)
        for key in anger_tree:
            for index, audio in enumerate(anger_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".avi"
                new_audio_path = join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = new_negative_index + index + 1

        # Process disgust
        disgust_tree = get_dataset(disgust_folder, 7)
        for key in disgust_tree:
            for index, audio in enumerate(disgust_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".avi"
                new_audio_path = join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)

        shutil.rmtree(sadness_folder)
        shutil.rmtree(fear_folder)
        shutil.rmtree(anger_folder)
        shutil.rmtree(disgust_folder)


def partition_folder_3_classes(root_path):
    # root_path should be train/valid/test
    group_positive(root_path)
    group_negative(root_path)

def partition_video_3classes(root_path):
    # root_path should be test_folder with train/valid/test
    organized_videos = get_dataset(root_path, 5)
    for key in organized_videos:
        partition_folder_3_classes(key)


def get_filename(filename, filenumber, index, interval):
    if index == 0:
        return filename + str(filenumber) + ".avi"
    number = (index * interval) + filenumber
    return filename + str(number) + ".avi"

def get_training_range(train_number, valid_number, test_number):
    train_number += 1
    valid_number = train_number + valid_number
    test_number = valid_number + test_number
    return train_number, valid_number, test_number

def move_file(source, destination, emotion, filenumber, filename, index, double = False):
    train_number, valid_number, test_number = get_training_range(TRAIN_VIDEO_NUMBER, VALID_VIDEO_NUMBER, TEST_VIDEO_NUMBER)
    if double:
        train_number = (train_number * 2) - 1
        valid_number = (valid_number * 2) - 1
        test_number = (test_number * 2) - 1
    if filenumber in range(1, train_number):
        name = get_filename(filename, filenumber, index, train_number - 1)
        shutil.copyfile(source, join(destination, "train\\", emotion, name))
    elif filenumber in range(train_number, valid_number):
        name = get_filename(filename, filenumber, index, valid_number - train_number)
        shutil.copyfile(source, join(destination, "valid\\", emotion, name))
    elif filenumber in range(valid_number, test_number):
        name = get_filename(filename, filenumber, index, test_number - valid_number)
        shutil.copyfile(source, join(destination, "test\\", emotion, name))



def prepare_for_prediction():
    pass










