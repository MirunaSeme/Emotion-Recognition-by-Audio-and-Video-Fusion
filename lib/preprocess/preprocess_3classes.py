from lib.constants.constants import constant_dir, category_3_folder
from lib.preprocess.preprocess_data import augment_with_overlap_dataset, process_spectrograms_dataset, extract_audio_from_video_dataset, process_spectrogram, augment_with_overlap, crop_spectrogram


if __name__ == '__main__':
    print(constant_dir["root"])
    # extract_audio_from_video_dataset(constant_dir["root"])
    # augment_with_overlap_dataset(constant_dir["root"], 6, 512, 128)
    # process_spectrograms_dataset(constant_dir["root"])
    # process_spectrogram("..\\..\\data\\3classes\\Tutu_with_pitch\\Tutu_with_pitch0.wav")
    # crop_spectrogram("..\\..\\data\\3classes\\Tutu_with_pitch\\Tutu_with_pitch0.png")
    # augment_with_overlap("..\\..\\data\\3classes\\Tutu_with_pitch\\n1.wav", 0, "..\\..\\data\\3classes\\Tutu_with_pitch", 512, 256, True)

