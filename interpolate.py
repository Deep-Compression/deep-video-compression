import os
import pickle
from pathlib import Path
import numpy as np

from frame_interpolation.interpolation import interpolate_frames_process
from frame_interpolation.linear import linear_interpolation
from frame_interpolation.sepconv_slomo import sepconv_slomo_interpolation

from helper.print_progress_bar import print_progress_bar
from helper.video_writer import write_frames_to_video

from config import *


def interpolate_frames():
    """
        Interpolates intermediate frames between decompressed key frames and writes generated videos to file.

        :param properties: Experiment properties
    """
    len_models, len_interpolation_depths = len(MODELS), len(INTERPOLATION_DEPTHS)
    
    # count sequences
    len_dataset_files = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        if 'im1.png' in files:
            len_dataset_files += 1
    
    process_steps = len_models * len_interpolation_depths * len_dataset_files

    for method in INTERPOLATION_METHODS:
        print('Interpolating frames using {} interpolation and evaluating the results...'.format(method))

        n = 0
        print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        if method == 'linear':
            interpolation_function = linear_interpolation

        elif method == 'sepconv_slomo':
            interpolation_function = sepconv_slomo_interpolation

        else:
            raise RuntimeError('Invalid interpolation method (\'{}\').'.format(method))

        for model in MODELS:
            for depth in INTERPOLATION_DEPTHS:
                key_frames_dir = OUTPUT_DIR + '/decompressed/key_frames/{}/depth_{}'.format(model, depth)

                for root, dirs, files in os.walk(DATASET_DIR):
                    if 'im1.png' in files:
                        output_path = OUTPUR_DIR + '/compressed/{}/depth_{}/'.format(model, depth)
                        key_frames_file = output_path + root.replace('/', '').replace('.', '') + '.keyframes'
                        key_frames_dict = pickle.load(open(key_frames_file, 'rb'))

                        frames = np.asarray(interpolate_frames_process(
                            key_frames=key_frames_dict['frames'],
                            num_end_frames=key_frames_dict['num_end_frames'],
                            method=interpolation_function,
                            depth=depth,
                            print_log_messages=False,
                            print_progress=False
                        ))

                        output_file = OUTPUR_DIR + '/decompressed/frames/{}/{}/depth_{}/' \
                            .format(method, model, depth) + root.replace('/', '').replace('.', '') + '.frames'
                        Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

                        pickle.dump({'frames': frames}, open(output_file, 'wb'))

                        output_file = OUTPUR_DIR + '/decompressed/videos/{}/{}/depth_{}/' \
                            .format(method, model, depth) + root.replace('/', '').replace('.', '') + '.mp4'
                        write_frames_to_video(output_file, frames, key_frames_dict['fps'], print_log_messages=False)

                        n += 1
                        print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        print('Interpolation of intermediate frames using {} interpolation complete!\n'.format(method))

