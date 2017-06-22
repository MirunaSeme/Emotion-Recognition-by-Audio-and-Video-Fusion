from lib.partitioning.partition_data_for_training import *
from lib.constants.constants import constant_dir, category_7_folder

if __name__ == '__main__':
    create_directories(constant_dir["training"], category_7_folder)
    prepare_for_train(constant_dir["training"], constant_dir["classes"])



