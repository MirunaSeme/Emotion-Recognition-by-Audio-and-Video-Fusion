import os
from lib.constants import flags

constant_dir = {}
AUGMENT = flags.AUGMENT
OVERLAP_SIZE = flags.OVERLAP_SIZE
OVERLAP_STEP = flags.OVERLAP_STEP

DATA = "..\\..\\data\\"
constant_dir["database"] = os.path.join(DATA, "database\\")

CLASS_NUMBER = '%sclasses\\' % flags.CLASS_NUMBER
CURRENT_DIR = os.path.join(DATA, CLASS_NUMBER, flags.TEST_NAME)
constant_dir["root"] = CURRENT_DIR

AUDIO_DIR = os.path.join(CURRENT_DIR, "audio\\")
SPECTROGRAM_DIR = os.path.join(CURRENT_DIR, "spectrogram\\")
constant_dir["audio"] = AUDIO_DIR
constant_dir["spectrogram"] = SPECTROGRAM_DIR


TRAIN = "ready_to_train\\"
CLASSES = "classes\\"
constant_dir["training"] = os.path.join(SPECTROGRAM_DIR, TRAIN)
constant_dir["classes"] = os.path.join(SPECTROGRAM_DIR, CLASSES)


training_folders = ["train", "valid", "test"]
category_7_folder = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
category_3_folder = ["negative", "neutral", "positive"]
data_folders = ["audio", "spectrogram"]

