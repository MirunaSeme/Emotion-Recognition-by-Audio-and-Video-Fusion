import os, subprocess
from os.path import basename, join, splitext, dirname
import math

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from pydub.pydub import AudioSegment

from PIL import Image

from lib.common.common import get_dataset, memory

resample_command = ""
divide_wav_command = ""
generate_spectogram_command = "find . -name *.wav -execdir sox -V1 {} -r 16000 {} ';'"

SAMPLE_PATH = "..\\data\\non-augmented\\samples\\"
AUGMENTED_SAMPLE_PATH = "..\\data\\augmented\\augmented_samples\\"
AUGMENTED_SAMPLE_WITH_OVERLAP_PATH = "..\\data\\augmented\\augmented_samples_with_overlap_2\\"
TEMP_AUGMENTED_SPECTS_PATH = "..\\data\\for_testing_augmented_spects\\"
TEMP_SPECTS_PATH = "..\\data\\for_testing_spects\\"
CROPPED_SPECTS_AUGMENTED_WITH_OVERLAP = "..\\data\\augmented\\cropped_augmented_with_overlap_spects_2\\"


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new frequencies
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i + 1]], axis=1)

    # make list of center of frequencies
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i + 1]])]

    return newspec, freqs


# Plot spectrograms: @Frank Zalkow
def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.savefig(plotpath, bbox_inches="tight")
    plt.clf()
    plt.close()


def get_number_of_splits(pydub_wavfile, split_size):
    length_pydub_wavfile = pydub_wavfile.__len__()
    return math.trunc(length_pydub_wavfile / split_size)


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
                augment_path = os.path.join(augment_root, os.path.basename(os.path.dirname(wavfile)),
                                            wavfile_split_name)
                wavfile_split.export(augment_path, format="wav")
                start += split_size
                finish += split_size
            print(wavfile, "[DONE]")


def augment_with_overlap_dataset(root_path, subtree, split_size, split_step, file_start_number=0, delete=True):
    samples_tree = get_dataset(root_path, subtree)
    for key in samples_tree:
        for wavfile in samples_tree[key]:
            file_start_number = augment_with_overlap(wavfile, file_start_number, key, split_size, split_step, delete)
            file_start_number += 1


def augment_with_overlap(filename, filename_number, root_directory, split_size, split_step, delete):
    pydub_wavfile = AudioSegment.from_wav(filename)
    number_of_splits = get_number_of_splits(pydub_wavfile, split_step) - 5
    start = split_size
    finish = 2*split_size
    for split in range(number_of_splits):
        wavfile_split = pydub_wavfile[start:finish]
        wavfile_split_name = basename(root_directory) + str(filename_number) + ".wav"
        filename_number += 1
        new_wavfile_path = os.path.join(os.path.dirname(filename), wavfile_split_name)
        wavfile_split.export(new_wavfile_path, format="wav")
        start += split_step
        finish += split_step
    print(filename, "[DONE]")
    # Remove previous wavfile
    if delete:
        os.remove(filename)
    return filename_number


def crop_spectrogram_dataset(path, subtree):
    spects_tree = get_dataset(path, subtree)
    for key in spects_tree:
        for spectrogram in spects_tree[key]:
            crop_spectrogram(spectrogram)


def crop_spectrogram(filename):
    img = Image.open(filename)
    new_img = img.crop((117, 13, 1025, 590))
    new_img.save(filename)
    print("Finished cropping", filename)


def spectrogram_dataset(path, subtree, delete=True):
    samples_tree = get_dataset(path, subtree)
    for key in samples_tree:
        wavfiles = [wavfile for wavfile in samples_tree[key] if ".wav" in wavfile]
        max_files = len(wavfiles)
        for wavfile in wavfiles:
            create_spectrogram(wavfile, delete, max_files)
            max_files -= 1


def create_spectrogram(filename, delete=True, file_num=None):
    spectfilename = os.path.basename(filename).split(".")[0] + ".png"
    plotpath = os.path.join(dirname(filename), spectfilename)

    plotstft(filename, plotpath=plotpath)

    print("[", file_num, "]", filename, "[DONE SPECTROGRAM] @ ", memory(), "GB")
    if delete:
        os.remove(filename)


def process_spectrograms_dataset(root_path, subtree=6, delete=True):
    spectrogram_dataset(root_path, subtree, delete)
    crop_spectrogram_dataset(root_path, subtree)


def process_spectrogram(filename):
    create_spectrogram(filename)
    crop_spectrogram(filename)


def extract_audio_from_video_dataset(root_path, subtree=6):
    directories = get_dataset(root_path, subtree)
    for directory_key in directories:
        for video in directories[directory_key]:
            extract_audio_from_video(video, directory_key)


def extract_audio_from_video(filename, directory, delete=True):
    audio_name = basename(splitext(filename)[0]) + ".wav"
    audio_file_path = join(directory, audio_name)
    command = "C:\\ffmpeg\\bin\\ffmpeg -loglevel quiet -i %s -f wav -ab 160k -vn %s" % (filename, audio_file_path)
    subprocess.call(command)
    if delete:
        os.remove(filename)
