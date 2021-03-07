"""
    Created on 18 Feb 2021 09:10am

    @author Felix Beutter
"""
import os
import cv2
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow debug output

from deep_video_compression.compress_video import compress_video
from deep_video_compression.decompress_dvc import decompress_dvc

from helper.split_video_in_clips import split_video_in_clips


def main():
    """
    dictionary = compress_video(input_file='dataset/raw_video.y4m',
                                output_file=None,
                                # interpolation='sepconv_slomo',
                                interpolation_depth=1,
                                model='hific-hi')

    pickle.dump(dictionary, open('dict.pickle', 'wb'))

    d = pickle.load(open('dict.pickle', 'rb'))
    decompress_dvc(d)
    """
    split_video_in_clips('resources/raw_video.y4m', 'dataset', 3, 24)


if __name__ == '__main__':
    main()
