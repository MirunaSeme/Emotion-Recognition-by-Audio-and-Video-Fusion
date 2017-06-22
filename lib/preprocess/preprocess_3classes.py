import os, shutil
from os.path import basename, splitext, dirname
from lib.constants.constants import constant_dir, category_3_folder, AUGMENT, OVERLAP_STEP, OVERLAP_SIZE
from lib.common.common import get_dataset, create_directory_tree
from lib.preprocess.preprocess_data import augment_dataset_with_overlap, spectrogram_dataset, crop_spectrogram

def group_positive(root_path):
    # Happiness & Surprise
    happiness_folder = os.path.join(root_path, "happiness\\")
    surprise_folder = os.path.join(root_path, "surprise\\")
    positive_folder = os.path.join(root_path, "positive\\")
    if not os.path.exists(positive_folder):
        os.mkdir(positive_folder)

        # Process happiness
        happiness_tree = get_dataset(happiness_folder, 7)
        for key in happiness_tree:
            for index, audio in enumerate(happiness_tree[key]):
                replaced_audio_name = basename(splitext(audio)[0]).replace('h', 'p') + ".wav"
                new_audio_path = os.path.join(dirname(positive_folder), replaced_audio_name)
                shutil.copy(audio, new_audio_path)
                # print(new_audio_path)
            new_positive_index = index + 2

        # Process surprise
        surprise_tree = get_dataset(surprise_folder, 7)
        for key in surprise_tree:
             for index, audio in enumerate(surprise_tree[key]):
                 replaced_audio_name = "p" + str(new_positive_index + index) + ".wav"
                 new_audio_path = os.path.join(dirname(positive_folder), replaced_audio_name)
                 shutil.move(audio, new_audio_path)
                 # print(new_audio_path)

        shutil.rmtree(happiness_folder)
        shutil.rmtree(surprise_folder)


def group_negative(root_path):
    # Sadness & Fear & Anger
    sadness_folder = os.path.join(root_path, "sadness\\")
    fear_folder = os.path.join(root_path, "fear\\")
    anger_folder = os.path.join(root_path, "anger\\")
    disgust_folder = os.path.join(root_path, "disgust\\")
    negative_folder = os.path.join(root_path, "negative\\")

    if not os.path.exists(negative_folder):
        os.mkdir(negative_folder)

        # Process sadness
        sadness_tree = get_dataset(sadness_folder, 7)
        for key in sadness_tree:
            for index, audio in enumerate(sadness_tree[key]):
                replaced_audio_name = basename(splitext(audio)[0]).replace('sa', 'neg') + ".wav"
                new_audio_path = os.path.join(dirname(negative_folder), replaced_audio_name)
                shutil.copy(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = index + 2

        # Process fear
        fear_tree = get_dataset(fear_folder, 7)
        for key in fear_tree:
            for index, audio in enumerate(fear_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".wav"
                new_audio_path = os.path.join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = new_negative_index + index + 1

        # Process anger
        anger_tree = get_dataset(anger_folder, 7)
        for key in anger_tree:
            for index, audio in enumerate(anger_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".wav"
                new_audio_path = os.path.join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)
            new_negative_index = new_negative_index + index + 1

        # Process disgust
        disgust_tree = get_dataset(disgust_folder, 7)
        for key in disgust_tree:
            for index, audio in enumerate(disgust_tree[key]):
                replaced_audio_name = "neg" + str(new_negative_index + index) + ".wav"
                new_audio_path = os.path.join(dirname(negative_folder), replaced_audio_name)
                shutil.move(audio, new_audio_path)
                # print(new_audio_path)

        shutil.rmtree(sadness_folder)
        shutil.rmtree(fear_folder)
        shutil.rmtree(anger_folder)
        shutil.rmtree(disgust_folder)


def process_audio(root_path):
    if not os.path.exists(root_path):
        shutil.copytree(constant_dir["database"], root_path)
        group_positive(root_path)
        group_negative(root_path)
        if AUGMENT:
            augment_dataset_with_overlap(root_path, 6, OVERLAP_SIZE, OVERLAP_STEP)

def process_spectrograms(root_path, audio_path):
    if not os.path.exists(root_path):
        create_directory_tree(root_path, category_3_folder)
        spectrogram_dataset(root_path, audio_path, 6)
        crop_spectrogram(root_path)
        directory_list = get_dataset(root_path, 6)
        if not os.path.exists(constant_dir["classes"]):
            os.mkdir(constant_dir["classes"])
        for dir_entry in directory_list:
            if(dir_entry != root_path):
                shutil.move(dir_entry, constant_dir["classes"])

if __name__ == '__main__':
    print(constant_dir["root_dir"])
    process_audio(constant_dir["audio"])
    process_spectrograms(constant_dir["spectrogram"], constant_dir["audio"])
