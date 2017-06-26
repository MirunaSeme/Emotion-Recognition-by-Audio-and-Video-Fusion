import os

# Generate directory
# Copy files from original to directory
# Recursively go through all the folders and generate the corresponding spectograms
# Delete wav files -. only spectograms remain
# Divide into train valid & sample (train & valid)

import os, shutil, subprocess
from os.path import basename, join, splitext, dirname
import math

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from pydub.pydub import AudioSegment

from pprint import pprint
from PIL import Image

from lib.common.common import get_dataset, create_training_directories
from lib.constants.constants import constant_dir


resample_command = ""
divide_wav_command = ""
generate_spectogram_command = "find . -name *.wav -execdir sox -V1 {} -r 16000 {} ';'"

SAMPLE_PATH="..\\data\\non-augmented\\samples\\"
AUGMENTED_SAMPLE_PATH="..\\data\\augmented\\augmented_samples\\"
AUGMENTED_SAMPLE_WITH_OVERLAP_PATH="..\\data\\augmented\\augmented_samples_with_overlap_2\\"
TEMP_AUGMENTED_SPECTS_PATH = "..\\data\\for_testing_augmented_spects\\"
TEMP_SPECTS_PATH = "..\\data\\for_testing_spects\\"
CROPPED_SPECTS_AUGMENTED_WITH_OVERLAP = "..\\data\\augmented\\cropped_augmented_with_overlap_spects_2\\"


# """ short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

# """ scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

# """ plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

def get_number_of_splits(pydub_wavfile, split_size):
    length_pydub_wavfile = pydub_wavfile.__len__()
    return math.trunc(length_pydub_wavfile/split_size)


def augment_dataset(split_size):
    samples_tree = get_dataset(SAMPLE_PATH)
    augment_root = AUGMENTED_SAMPLE_PATH
    for key in samples_tree:
        for wavfile in samples_tree[key]:
            pydub_wavfile = AudioSegment.from_wav(wavfile)
            number_of_splits = get_number_of_splits(pydub_wavfile, split_size)
            start = 0
            finish = split_size
            for split in range(number_of_splits):
                wavfile_split = pydub_wavfile[start:finish]
                wavfile_split_name = os.path.basename(wavfile).split(".")[0] + str(split) + ".wav"
                augment_path = os.path.join(augment_root, os.path.basename(os.path.dirname(wavfile)), wavfile_split_name)
                print(augment_path)
                wavfile_split.export(augment_path, format="wav")
                start += split_size
                finish += split_size
            print(wavfile, "[DONE]")

def augment_dataset_with_overlap(root_path, subtree, split_size, split_step):
    samples_tree = get_dataset(root_path, subtree)
    print(samples_tree)
    for key in samples_tree:
        wavfile_number = 0
        for wavfile in samples_tree[key]:
            pydub_wavfile = AudioSegment.from_wav(wavfile)
            # Number 3 = pair of two x start x finish
            # Number 4 = pair of two x start x finish x another because split is smaller.
            # It would have been 5, but eliminated one because changes start to bigger value (not split_step)
            number_of_splits = get_number_of_splits(pydub_wavfile, split_step) - 9
            start = 512
            finish = split_size + split_step
            for split in range(number_of_splits):
                wavfile_split = pydub_wavfile[start:finish]
                wavfile_split_name = basename(key) + str(wavfile_number) + ".wav"
                wavfile_number += 1
                new_wavfile_path = os.path.join(os.path.dirname(wavfile), wavfile_split_name)
                # print(new_wavfile_path)
                wavfile_split.export(new_wavfile_path, format="wav")
                start += split_step
                finish += split_step
            print(wavfile, "[DONE]")
            # Remove previous wavfile
            os.remove(wavfile)


def crop_spectrogram(path,subtree):
    spects_tree = get_dataset(path, subtree)
    for key in spects_tree:
        for spectrogram in spects_tree[key]:
            img = Image.open(spectrogram)
            new_img = img.crop((117, 13, 1025, 590))
            new_img.save(spectrogram)
            print("Finished cropping", spectrogram)

def spectrogram_dataset(path, subtree):
    samples_tree = get_dataset(path, subtree)
    for key in samples_tree:
        for wavfile in samples_tree[key]:
            spectfilename = os.path.basename(wavfile).split(".")[0] + ".png"
            plotpath = os.path.join(dirname(wavfile), spectfilename)
            plotstft(wavfile, plotpath=plotpath)
            print(wavfile, "[DONE SPECTROGRAM]")
            os.remove(wavfile)

def process_spectrograms(root_path):
    spectrogram_dataset(root_path, 6)
    crop_spectrogram(root_path, 6)

def extract_audio_from_video(root_path):
    directories = get_dataset(root_path, 6)
    for directory_key in directories:
        for video in directories[directory_key]:
            audio_name = basename(splitext(video)[0]) + ".wav"
            audio_file_path = join(directory_key, audio_name)
            command = "C:\\ffmpeg\\bin\\ffmpeg -loglevel quiet -i %s -f wav -ab 160k -vn %s" % (video, audio_file_path)
            subprocess.call(command)
            os.remove(video)
        print("Finished: " + directory_key)





