import pickle
import os
from pathlib import Path

import cv2

from image_compression.hific_compression import compress_process


def compress_sequence(input_path, output_file='compressed_video.dvc', model='hific-lo', interpolation_depth=1,
                   print_log_messages=True, print_progress=True):
    """
        Compresses a video file using the hific generative image compression.

        :param input_path: Path to sequence to compress
        :param output_file: File name of compressed sequence
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param interpolation_depth: Determines the number of frames to interpolate between to compressed ones
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Dictionary including packed tensors of the compressed video and compression properties
    """
    # load frames into array
    sequence = []
    file_list = os.listdir(input_path)
    for image in file_list:
        if len(os.path.basename(image)) == 7:
            sequence.append(cv2.imread(image))

    num_frames = len(sequence)
    num_end_frames = (num_frames - 1) % (2 ** interpolation_depth) + 1

    frames_to_compress = []
    for n, frame in enumerate(sequence):
        if n % (2 ** interpolation_depth) == 0 or n >= num_frames - num_end_frames:
            frames_to_compress.append(frame)

    if num_frames < 1:
        raise RuntimeError('Sequence has no images (\'{}\').'.format(input_path))

    dictionary = {'fps': 30, 'model': model, 'num_frames': num_frames, 'interpolation_depth': interpolation_depth,
                  'num_end_frames': num_end_frames}

    packed_tensors = compress_process(frames_to_compress, model, print_log_messages, print_progress)
    dictionary['packed_tensors'] = packed_tensors

    if output_file is not None:
        Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
        pickle.dump(dictionary, open(output_file, 'wb'))

    return dictionary
