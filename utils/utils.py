import os

def check_dir_creation(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)