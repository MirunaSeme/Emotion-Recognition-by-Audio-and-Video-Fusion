from lib.partitioning.partition_data_for_training import *
from lib.constants.constants import constant_dir, category_3_folder
from lib.common.common import create_training_directories


if __name__ == '__main__':
    # Path to sample audio for 7 classes
    # If sample path does not exist, create it
    create_training_directories(constant_dir["training"], category_3_folder)
    prepare_for_train(constant_dir["training"], constant_dir["classes"], 7)

