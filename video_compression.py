import math
import os
import pickle
import sys
from multiprocessing import Process, Manager

import cv2
import numpy as np
import torch

from sepconv_slomo.run import estimate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from tf_compression.models.tfci import import_metagraph, instantiate_signature


def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='█',
                       end='\r'):
    """
        Prints a progress bar to console.

        :param iteration: Current iteration
        :param total: Total iterations
        :param prefix: Prefix string
        :param suffix: Suffix string
        :param decimals: Number of decimals in percent complete
        :param length: Character length
        :param fill: Bar fill character
        :param end: End character (e.g. '\r', '\r\n')
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=end)

    if iteration == total:
        print()


def compress_hific(frames, model='hific-lo'):
    """
        Compresses an image using a trained hific model.

        :param frames: Frames to compress (numpy arrays)
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :return: Packed tensors of the compressed images
    """
    packed_tensors = []
    print_progress_bar(0, len(frames))

    with tf.Graph().as_default():
        if model not in ['hific-lo', 'hific-mi', 'hific-hi']:
            raise RuntimeError('Invalid hific model name \'' + model + '\'!')

        signature_defs = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_defs['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        with tf.Session() as session:
            for i, frame in enumerate(frames):
                frame = np.expand_dims(frame, 0)
                arrays = session.run(outputs, feed_dict={inputs: frame})

                packed = tfc.PackedTensors()
                packed.pack(outputs, arrays)

                packed_tensors.append(packed.string)

                print_progress_bar(i + 1, len(frames))

    return packed_tensors


def decompress_hific(packed_tensors, model, process_dict=None):
    """
        Decompresses a list of packed tensors of images compressed with a hific network.

        :param packed_tensors: Packed tensors
        :param model: Hific model used for image compression
        :param process_dict: Dictionary to store return value if multiprocessing is used
        :return: Decompressed frames (as numpy arrays)
    """
    frames = []

    print('Decompressing packed tensors using hific')
    print_progress_bar(0, len(packed_tensors))

    with tf.Graph().as_default():
        signature_defs = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_defs['receiver'])

        inputs = [inputs[k] for k in sorted(inputs) if k.startswith('channel:')]

        with tf.Session() as session:
            for i, packed_tensor in enumerate(packed_tensors):
                arrays = tfc.PackedTensors(packed_tensor).unpack(inputs)
                frame = session.run(outputs['output_image'], feed_dict=dict(zip(inputs, arrays)))

                frame = np.squeeze(frame, 0)
                frame = np.round(frame)
                frame = np.clip(frame, 0, 255)
                frame = frame.astype(np.uint8)

                frames.append(frame)

                print_progress_bar(i + 1, len(packed_tensors))

    if process_dict is not None:
        process_dict['result'] = frames

    return frames


def linear_interpolation(first_frame, last_frame, num_intermediate_frames=1):
    """
        Generates intermediate frames between a first and last frame using linear interpolation.

        :param first_frame: First frame
        :param last_frame: Last frame
        :param num_intermediate_frames: Number of intermediate frames to be generated
        :return: Numpy array of generated intermediate frames
    """
    return np.linspace(first_frame, last_frame, num_intermediate_frames + 2, dtype=np.uint8)[1:-1]


def sepconv_slomo_interpolation(first_frame, second_frame):
    """
        Generates one intermediate frame between two frames using sepconv slomo.

        :param first_frame: First frame
        :param second_frame: Second frame
        :return: Interpolated intermediate frames
    """
    first_frame_data = np.array(first_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    first_tensor = torch.FloatTensor(np.ascontiguousarray(first_frame_data))

    second_frame_data = np.array(second_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    second_tensor = torch.FloatTensor(np.ascontiguousarray(second_frame_data))

    output = estimate(first_tensor, second_tensor)
    return (output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)


def recursive_sepconv_slomo_interpolation(first_frame, second_frame, num_recursion_steps):
    """
        Uses the sepconv slomo interpolation to recursively interpolate intermediate images between two given frames.

        :param first_frame: First frame
        :param second_frame: Second frame
        :param num_recursion_steps: Decides the number of intermediate frames to be generated
        :return: List of interpolated intermediate frames
    """
    frames = [first_frame, second_frame]

    for _ in range(num_recursion_steps):
        intermediate_frames = []

        for i in range(len(frames) - 1):
            intermediate_frames.append(sepconv_slomo_interpolation(frames[i], frames[i + 1]))

        merged_frames = [None] * (len(frames) + len(intermediate_frames))
        merged_frames[::2] = frames
        merged_frames[1::2] = intermediate_frames

        frames = merged_frames

    return frames[1:-1]


def compress(input_file, output_file='compressed_video.dvc', model='hific-lo', interpolation=None,
             num_intermediate_frames=1):
    """
        Compresses a video file using the hific generative image compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param interpolation: Method to interpolate frames for further data reduction (None for no interpolation)
        :param num_intermediate_frames: Number of frames to interpolate between to compressed ones
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_frames = 1000

    if num_frames < 1:
        print('ERROR: Video has no frames, compression will not be performed')
        exit(0)

    print("Starting compression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    num_end_frames = (num_frames - 1) % (num_intermediate_frames + 1) + 1

    dictionary = {'fps': fps, 'model': model, 'num_frames': num_frames, 'interpolation': interpolation,
                  'num_intermediate_frames': num_intermediate_frames, 'num_end_frames': num_end_frames}

    frames_to_compress = []
    _, cv_image = video_capture.read()

    for n in range(num_frames):
        if interpolation is None or n % (num_intermediate_frames + 1) == 0 or n >= num_frames - num_end_frames:
            frames_to_compress.append(cv_image)

        _, cv_image = video_capture.read()

    compressed_frames = compress_hific(frames_to_compress, model)
    dictionary['compressed_frames'] = compressed_frames

    pickle.dump(dictionary, open(output_file, 'wb'))


def decompress(input_file, output_file='decompressed_video.mp4'):
    """
        Decompresses a compressed video file using the hific generative image compression (and frame interpolation if
        applicable)

        :param input_file: Compressed video to decompress
        :param output_file: File name of decompressed video
    """
    dictionary = pickle.load(open(input_file, 'rb'))
    print("Starting decompression of file '" + input_file + "' (" + str(dictionary['num_frames']) + ' frames)...')

    num_end_frames = dictionary['num_end_frames']
    interpolation = dictionary['interpolation']
    num_intermediate_frames = dictionary['num_intermediate_frames']

    compressed_frames = dictionary['compressed_frames']

    manager = Manager()
    process_dict = manager.dict()

    decompression_process = Process(target=decompress_hific,
                                    args=[compressed_frames, dictionary['model'], process_dict])
    decompression_process.start()
    decompression_process.join()

    decompressed_frames = process_dict['result']

    if interpolation is None:
        frames = decompressed_frames

    else:
        frames = []

        print('Interpolating intermediate frames')
        print_progress_bar(0, len(decompressed_frames) - num_end_frames)

        for n in range(len(decompressed_frames) - num_end_frames):
            frames.append(decompressed_frames[n])

            if interpolation == 'sepconv_slomo':
                recursion_depth = int(math.log(num_intermediate_frames + 1, 2))
                frames.extend(recursive_sepconv_slomo_interpolation(decompressed_frames[n], decompressed_frames[n + 1],
                                                                    recursion_depth))
            else:
                frames.extend(
                    linear_interpolation(decompressed_frames[n], decompressed_frames[n + 1], num_intermediate_frames))

            print_progress_bar(n + 1, len(decompressed_frames) - num_end_frames)

        frames.extend(decompressed_frames[-num_end_frames:])

    frame_shape = np.shape(frames[0])
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), dictionary['fps'], (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


def main():
    '''
    if sys.argv[1] == 'compress':
        compress(*sys.argv[2:])

    else:
        decompress(*sys.argv[2:])
    '''

    # compress('video_raw.y4m', interpolation='sepconv_slomo', num_intermediate_frames=3)
    # compress('image.mp4')
    decompress('compressed_video.dvc')


if __name__ == '__main__':
    main()
