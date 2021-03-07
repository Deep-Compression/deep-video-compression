"""
    Created on 18 Feb 2021 09:42am

    @author Felix Beutter
"""
import pickle
from pathlib import Path

import cv2

from image_compression.hific_compression import compress_process


def compress_video(input_file, output_file='compressed_video.dvc', model='hific-lo', interpolation_depth=1,
                   print_log_messages=True, print_progress=True):
    """
        Compresses a video file using the hific generative image compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param interpolation_depth: Determines the number of frames to interpolate between to compressed ones
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Dictionary including packed tensors of the compressed video and compression properties
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < 1:
        raise RuntimeError('Video has no frames, compression will not be performed (\'{}\').'.format(input_file))

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    num_end_frames = (num_frames - 1) % (2 ** interpolation_depth) + 1

    dictionary = {'fps': fps, 'model': model, 'num_frames': num_frames, 'interpolation_depth': interpolation_depth,
                  'num_end_frames': num_end_frames}

    frames_to_compress = []
    _, cv_image = video_capture.read()

    for n in range(num_frames):
        if n % (2 ** interpolation_depth) == 0 or n >= num_frames - num_end_frames:
            frames_to_compress.append(cv_image)

        _, cv_image = video_capture.read()

    packed_tensors = compress_process(frames_to_compress, model, print_log_messages, print_progress)
    dictionary['packed_tensors'] = packed_tensors

    if output_file is not None:
        Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
        pickle.dump(dictionary, open(output_file, 'wb'))

    return dictionary
