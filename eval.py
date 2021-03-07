"""
    Created on 18 Feb 2021 15:56pm

    @author Felix Beutter
"""
import os
import pickle
from pathlib import Path

import numpy as np
from terminaltables import AsciiTable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow debug output

from frame_interpolation.interpolation import interpolate_frames_process
from frame_interpolation.linear import linear_interpolation
from frame_interpolation.sepconv_slomo import sepconv_slomo_interpolation
from helper.video_writer import write_frames_to_video
from image_compression.hific_decompression import decompress_process

from deep_video_compression.compress_video import compress_video
from metrics.multi_scale_ssim import multi_scale_ssim
from metrics.psnr import calculate_psnr
from metrics.PerceptualSimilarity.lpips import calculate_lpips
from helper.video_to_frames import video_to_frames
from helper.print_progress_bar import print_progress_bar


class ExperimentProperties:
    """
        Contains the properties for a test/evaluation experiment.
    """

    def __init__(self, dataset_dir='dataset', output_dir='output', models=['hific-hi'],
                 interpolation_methods=['linear', 'sepconv_slomo'], interpolation_depths=[1, 3]):
        """
            :param dataset_dir: Dataset directory
            :param output_dir: Output directory
            :param models: HiFiC models for compression
            :param interpolation_methods: Interpolation methods
            :param interpolation_depths: Recursion depth for interpolating intermediate frames
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.models = models
        self.interpolation_methods = interpolation_methods
        self.interpolation_depths = interpolation_depths

        if not os.path.isdir(dataset_dir):
            raise RuntimeError('Dataset directory does not exist!')

        self.dataset_files = os.listdir(dataset_dir)


def compress_dataset(properties):
    """
        Compresses dataset and writes results to file. Evaluation of compression factor will be printed.

        :param properties: Experiment properties
    """
    print('Compression of dataset files...')

    len_models, len_interpolation_depths, len_dataset_files = len(properties.models), len(
        properties.interpolation_depths), len(properties.dataset_files)

    compression_results = np.zeros((len_models, len_interpolation_depths, len_dataset_files))
    process_steps = len_models * len_interpolation_depths * len_dataset_files

    n = 0
    print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

    for i, model in enumerate(properties.models):
        for j, depth in enumerate(properties.interpolation_depths):
            for k, file in enumerate(properties.dataset_files):
                input_file = properties.dataset_dir + '/' + file

                output_path = properties.output_dir + '/compressed/{}/depth_{}/'.format(model, depth)
                output_file = output_path + os.path.splitext(file)[0] + '.dvc'

                compress_video(
                    input_file=input_file,
                    output_file=output_file,
                    model=model,
                    interpolation_depth=depth,
                    print_log_messages=False,
                    print_progress=False
                )

                original_file_size = os.path.getsize(input_file)
                compressed_file_size = os.path.getsize(output_file)
                compression_results[i][j][k] = compressed_file_size / original_file_size

                n += 1
                print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

    compression_results = np.mean(compression_results * 100, axis=-1).astype('str')

    properties.models.insert(0, '')
    compression_results = np.insert(compression_results, 0, properties.interpolation_depths, axis=0)
    compression_results = np.insert(compression_results, 0, properties.models, axis=1)

    print('\nCompression complete! Results (compression factors/file size ratio):')
    table = AsciiTable(compression_results.tolist())
    print(table.table + '\n')

    properties.models.pop(0)


def decompress_dataset(properties):
    """
        Decompresses compressed video files and writes the decompressed key frames to file (no interpolation).

        :param properties: Experiment properties
    """
    print('Decompression of dataset key frames...')

    len_models, len_interpolation_depths, len_dataset_files = len(properties.models), len(
        properties.interpolation_depths), len(properties.dataset_files)
    process_steps = len_models * len_interpolation_depths * len_dataset_files

    n = 0
    print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

    for i, model in enumerate(properties.models):
        for j, depth in enumerate(properties.interpolation_depths):
            compressed_files_dir = properties.output_dir + '/compressed/{}/depth_{}'.format(model, depth)

            for k, file in enumerate(properties.dataset_files):
                input_file = compressed_files_dir + '/' + os.path.splitext(file)[0] + '.dvc'
                output_file = properties.output_dir + '/decompressed/key_frames/{}/depth_{}/'. \
                    format(model, depth) + os.path.splitext(file)[0] + '.keyframes'

                dictionary = pickle.load(open(input_file, 'rb'))
                decompressed_frames = decompress_process(
                    packed_tensors=dictionary['packed_tensors'],
                    model=dictionary['model'],
                    print_log_messages=False,
                    print_progress=False
                )

                Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
                pickle.dump({'frames': decompressed_frames, 'num_end_frames': dictionary['num_end_frames'],
                             'fps': dictionary['fps']}, open(output_file, 'wb'))

                n += 1
                print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

    print('Decompression complete!\n')


def interpolate_frames(properties):
    """
        Interpolates intermediate frames between decompressed key frames and writes generated videos to file.

        :param properties: Experiment properties
    """
    len_models, len_interpolation_depths, len_dataset_files = len(properties.models), len(
        properties.interpolation_depths), len(properties.dataset_files)
    process_steps = len_models * len_interpolation_depths * len_dataset_files

    for method in properties.interpolation_methods:
        print('Interpolating frames using {} interpolation and evaluating the results...'.format(method))

        n = 0
        print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        if method == 'linear':
            interpolation_function = linear_interpolation

        elif method == 'sepconv_slomo':
            interpolation_function = sepconv_slomo_interpolation

        else:
            raise RuntimeError('Invalid interpolation method (\'{}\').'.format(method))

        for model in properties.models:
            for depth in properties.interpolation_depths:
                key_frames_dir = properties.output_dir + '/decompressed/key_frames/{}/depth_{}'.format(model, depth)

                for file in properties.dataset_files:
                    key_frames_file = key_frames_dir + '/' + os.path.splitext(file)[0] + '.keyframes'
                    key_frames_dict = pickle.load(open(key_frames_file, 'rb'))

                    frames = np.asarray(interpolate_frames_process(
                        key_frames=key_frames_dict['frames'],
                        num_end_frames=key_frames_dict['num_end_frames'],
                        method=interpolation_function,
                        depth=depth,
                        print_log_messages=False,
                        print_progress=False
                    ))

                    output_file = properties.output_dir + '/decompressed/frames/{}/{}/depth_{}/' \
                        .format(method, model, depth) + os.path.splitext(file)[0] + '.frames'
                    Path('/'.join(output_file.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)

                    pickle.dump({'frames': frames}, open(output_file, 'wb'))

                    output_file = properties.output_dir + '/decompressed/videos/{}/{}/depth_{}/' \
                        .format(method, model, depth) + os.path.splitext(file)[0] + '.mp4'
                    write_frames_to_video(output_file, frames, key_frames_dict['fps'], print_log_messages=False)

                    n += 1
                    print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        print('Interpolation of intermediate frames using {} interpolation complete!\n'.format(method))


def evaluate(properties):
    """
        Evaluates the decompressed videos using MS-SSIM.

        :param properties: Experiment properties
    """
    len_models, len_interpolation_depths, len_dataset_files = len(properties.models), len(
        properties.interpolation_depths), len(properties.dataset_files)
    process_steps = len_models * len_interpolation_depths * len_dataset_files

    for method in properties.interpolation_methods:
        print('Evaluating decompressed keyframes and interpolated intermediate frames ({} interpolation)...'.format(
            method))
        results = np.zeros((len_models, len_interpolation_depths, len_dataset_files))

        n = 0
        print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        for i, model in enumerate(properties.models):
            for j, depth in enumerate(properties.interpolation_depths):
                frames_dir = properties.output_dir + '/decompressed/frames/{}/{}/depth_{}'.format(method, model, depth)

                for k, file in enumerate(properties.dataset_files):
                    frames_file = frames_dir + '/' + os.path.splitext(file)[0] + '.frames'
                    frames_dict = pickle.load(open(frames_file, 'rb'))

                    original_frames = np.array(video_to_frames(properties.dataset_dir + '/' + file))
                    results[i][j][k] = multi_scale_ssim(original_frames, frames_dict['frames'])
                    #results[i][j][k] = calculate_psnr(img0, img1)
                    #results[i][j][k] = calculate_lpips(img0, img1)

                    n += 1
                    print_progress_bar(n, process_steps, suffix='({}/{} files)'.format(n, process_steps))

        results = np.mean(results, axis=-1).astype('str')

        properties.models.insert(0, '')
        results = np.insert(results, 0, properties.interpolation_depths, axis=0)
        results = np.insert(results, 0, properties.models, axis=1)

        print('\nEvaluation complete! Results (mean MS-SSIM):')
        table = AsciiTable(results.tolist())
        print(table.table + '\n')

        properties.models.pop(0)


def main():
    properties = ExperimentProperties()

    compress_dataset(properties)
    decompress_dataset(properties)
    interpolate_frames(properties)
    evaluate(properties)


if __name__ == '__main__':
    main()
