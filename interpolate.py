import os
import pickle
from builtins import len
from pathlib import Path

import cv2
import numpy as np

from config import *
from frame_interpolation.linear import linear_interpolation
from frame_interpolation.sepconv_slomo import sepconv_slomo_interpolation
from helper.print_progress_bar import print_progress_bar

for method in INTERPOLATION_METHODS:
    print('Interpolation of decompressed files using ' + method + '...')

    if method == 'linear':
        interpolation_function = linear_interpolation

    elif method == 'sepconv_slomo':
        interpolation_function = sepconv_slomo_interpolation

    else:
        raise RuntimeError('Invalid interpolation method \'{}\'.'.format(method))

    for depth in INTERPOLATION_DEPTHS:
        print('Interpolation with ' + str(2 ** depth - 1) + ' intermediate frames...')

        n = 0
        print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

        if depth not in [1, 2]:
            raise Exception('Interpolation depth too low or too high.')

        if depth == 1:
            key_frame_indices = [1, 3, 5, 7]

        else:
            key_frame_indices = [2, 6]

        for root, _, files in os.walk(DECOMPRESSED_DIR):
            if 'im1.png' in files:
                out_root = INTERPOLATED_DIR + root[21:]

                frames = []

                for i in range(len(key_frame_indices) - 1):
                    first_frame = cv2.imread(root + '/im' + str(key_frame_indices[i]) + '.png')
                    first_frame = np.expand_dims(first_frame, 0)

                    second_frame = cv2.imread(root + '/im' + str(key_frame_indices[i + 1]) + '.png')
                    second_frame = np.expand_dims(second_frame, 0)

                    intermediate_frames = interpolation_function(first_frame, second_frame, depth)

                    frames.append(first_frame)
                    frames.extend(intermediate_frames)

                frames.append(second_frame)

                for j, frame in enumerate(frames):
                    out_path = out_root + '/depth_' + str(depth) + '/im' + str(j + 1) + '.png'
                    Path('/'.join(out_path.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

                    frame = np.squeeze(frame, 0)
                    cv2.imwrite(out_path, frame)

                n += 1
                print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

        print()
    print()
