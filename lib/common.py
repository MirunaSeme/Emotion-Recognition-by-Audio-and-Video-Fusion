import os
import math
from pydub.pydub import AudioSegment
from PIL import Image

def get_dataset(path):
    samples_tree = {}
    for root, dirs, files in os.walk(path):
        path = root.split(os.sep)
        if len(path)-1 == 4:
            samples_tree[root] = []
        print((len(path)-1), (len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', os.path.join(root,file))
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

def write_size_spectrograms(path, filename):
    target = open(filename, 'w')
    spects_tree = get_dataset(path)
    for key in spects_tree:
        for spectogram in spects_tree[key]:
            img = Image.open(spectogram)
            target.write(str(img.size))
            target.write("\n")
    target.close()