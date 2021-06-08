import os
from builtins import len
from pathlib import Path
import cv2

from config import *
from frame_interpolation.linear import linear_interpolation
from frame_interpolation.sepconv_slomo import sepconv_slomo_interpolation
from frame_interpolation.rife import rife_interpolation
from helper.print_progress_bar import print_progress_bar

num_sequences = 1000

"""
for _, _, files in os.walk(DATASET_DIR):
    if 'im1.png' in files:
        num_sequences += 1
"""

for method in ['sepconv_slomo']:
    print('Interpolation of jpeg files using ' + method + '...')

    if method == 'linear':
        interpolation_function = linear_interpolation

    elif method == 'sepconv_slomo':
        interpolation_function = sepconv_slomo_interpolation

    elif method == 'rife':
        interpolation_function = rife_interpolation

    else:
        raise RuntimeError('Invalid interpolation method \'{}\'.'.format(method))

    for depth in INTERPOLATION_DEPTHS:
        print('Interpolation with ' + str(2 ** depth - 1) + ' intermediate frames...')

        if depth not in [1, 2]:
            raise Exception('Interpolation depth too low or too high.')

        if depth == 1:
            key_frame_indices = [1, 3, 5, 7]

        else:
            key_frame_indices = [2, 6]

        for model in MODELS:
            print('Interpolation with JPEG images based on {} dc file bitrates...'.format(model))

            n = 0
            print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

            for root, _, files in os.walk('./output/jpeg/' + model):
                if 'im1.jpg' in files:
                    out_root = INTERPOLATED_DIR + '/' + method + '/depth_' + str(depth) + '/' + model + root[22:]

                    frames = []

                    for i in range(len(key_frame_indices) - 1):
                        first_frame = cv2.imread(root + '/im' + str(key_frame_indices[i]) + '.jpg')
                        second_frame = cv2.imread(root + '/im' + str(key_frame_indices[i + 1]) + '.jpg')

                        intermediate_frames = interpolation_function(first_frame, second_frame, depth)

                        frames.append(first_frame)
                        frames.extend(intermediate_frames)

                    frames.append(second_frame)

                    for j, frame in enumerate(frames):
                        out_path = out_root + '/im' + str(j + 1) + '.png'
                        Path('/'.join(out_path.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(out_path, frame)

                    n += 1
                    print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

                    if n == num_sequences:
                        break

        print()
    print()
