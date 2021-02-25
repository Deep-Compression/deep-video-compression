"""
    Created on 18 Feb 2021 09:51am

    @author Felix Beutter
"""
import pickle

import cv2
import numpy as np

from .frame_interpolation import interpolate_frames, linear_interpolation, sepconv_slomo_interpolation
from .helper import print_progress_bar
from .image_compression import decompress_tensors_hific


def decompress_dvc(input_file, output_file='decompressed_video.mp4'):
    """
        Decompresses a compressed video file using the hific generative image compression (and frame interpolation if
        applicable)

        :param input_file: Compressed video to decompress
        :param output_file: File name of decompressed video
    """
    dictionary = pickle.load(open(input_file, 'rb'))

    packed_tensors = dictionary['packed_tensors']
    model = dictionary['model']
    interpolation = dictionary['interpolation']

    decompressed_frames = decompress_tensors_hific(packed_tensors, model)

    if interpolation is None:
        frames = decompressed_frames

    else:
        print('Interpolating intermediate frames...')

        num_end_frames = dictionary['num_end_frames']
        print_progress_bar(0, len(decompressed_frames) - num_end_frames)

        frames = []
        interpolation_depth = dictionary['interpolation_depth']

        for n in range(len(decompressed_frames) - num_end_frames):
            frames.append(decompressed_frames[n])

            if interpolation == 'linear':
                interpolated_frames = interpolate_frames(decompressed_frames[n], decompressed_frames[n + 1],
                                                         linear_interpolation, interpolation_depth)

            elif interpolation == 'sepconv_slomo':
                interpolated_frames = interpolate_frames(decompressed_frames[n], decompressed_frames[n + 1],
                                                         sepconv_slomo_interpolation, interpolation_depth)

            else:
                raise RuntimeError('Invalid interpolation method (\'{}\').'.format(interpolation))

            frames.extend(interpolated_frames)
            print_progress_bar(n + 1, len(decompressed_frames) - num_end_frames)

        frames.extend(decompressed_frames[-num_end_frames:])

    frame_shape = np.shape(frames[0])
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), dictionary['fps'], (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
