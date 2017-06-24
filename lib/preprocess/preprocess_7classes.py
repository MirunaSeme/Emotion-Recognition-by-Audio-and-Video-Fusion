from lib.preprocess.preprocess_data import augment_dataset_with_overlap, process_spectrograms, extract_audio_from_video
from lib.constants.constants import constant_dir, category_7_folder

if __name__ == '__main__':
    print(constant_dir["root"])
    extract_audio_from_video(constant_dir["root"])
    augment_dataset_with_overlap(constant_dir["root"], 6, 512, 256)
    process_spectrograms(constant_dir["spectrogram"], constant_dir["root"], category_7_folder)
