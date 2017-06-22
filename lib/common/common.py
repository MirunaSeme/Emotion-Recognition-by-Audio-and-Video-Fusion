import os, shutil
import math
from pydub.pydub import AudioSegment
from PIL import Image
from lib.constants.constants import training_folders

def get_dataset(path, subtree):
    samples_tree = {}
    for root, dirs, files in os.walk(path):
        path = root.split(os.sep)
        if len(path)-1 == subtree:
            samples_tree[root] = []
        # print((len(path)-1), (len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            # print(len(path) * '---', os.path.join(root,file))
            if root in samples_tree:
                if os.path.isfile(os.path.join(root,file)):
                    samples_tree[root].append(os.path.join(root,file))
    return samples_tree

def split_list(a_list, measure=1):
    boundary = math.trunc(len(a_list)*measure)
    return a_list[:boundary], a_list[boundary:]

def get_size_wavfile(wavfile):
    pydub_wavfile = AudioSegment.from_wav(wavfile)
    return pydub_wavfile.__len__()

def write_size_wavfile(path, filename):
    target = open(filename, 'w')
    audio_sample_tree = get_dataset(path)
    for key in audio_sample_tree:
        for wavfile in audio_sample_tree[key]:
            target.write(str(get_size_wavfile(wavfile)))
            target.write("\n")
    target.close()

def write_size_spectrograms(path, filename, subtree):
    target = open(filename, 'w')
    spects_tree = get_dataset(path, subtree)
    for key in spects_tree:
        for spectogram in spects_tree[key]:
            img = Image.open(spectogram)
            target.write(str(img.size))
            target.write("\n")
    target.close()

def create_training_directories(root_path, category_folder):
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for folder in training_folders:
        for category in category_folder:
            os.makedirs(os.path.join(root_path, folder, category), exist_ok=True)

def create_directory_tree(root_path, category_folder):
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for category in category_folder:
        os.makedirs(os.path.join(root_path, category), exist_ok=True)