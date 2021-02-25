"""
    Created on 18 Feb 2021 10:53am

    @author Felix Beutter
"""
import numpy as np
from multiprocessing import Manager, Process

import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from image_compression.hific_helper import import_metagraph, instantiate_signature
from helper.print_progress_bar import print_progress_bar


def compress(frames, model='hific-lo', process_dict=None, print_log_messages=True, print_progress=True):
    """
        Compresses a list of frames using a trained hific model.

        :param frames: Frames to compress (numpy arrays)
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param process_dict: Process dictionary to store results if method is executed in isolated process
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Packed tensors of the compressed images
    """
    if print_log_messages:
        print('Starting compression of frames using HiFiC ({} frames)...'.format(len(frames)))

    packed_tensors = []

    if print_progress:
        print_progress_bar(0, len(frames))

    with tf.Graph().as_default():
        if model not in ['hific-lo', 'hific-mi', 'hific-hi']:
            raise RuntimeError('Invalid hific model name \'' + model + '\'!')

        signature_def = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_def['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        with tf.Session() as session:
            for i, frame in enumerate(frames):
                frame = np.expand_dims(frame, 0)
                arrays = session.run(outputs, feed_dict={inputs: frame})

                packed = tfc.PackedTensors()
                packed.pack(outputs, arrays)

                packed_tensors.append(packed.string)

                if print_progress:
                    print_progress_bar(i + 1, len(frames))

    if process_dict is not None:
        process_dict['return'] = packed_tensors

    return packed_tensors


def compress_process(frames, model='hific-lo', print_log_messages=True, print_progress=True):
    """
        Compresses a list of frames using a trained hific model (executed in an isolated process).

        :param frames: Frames to compress (numpy arrays)
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Packed tensors of the compressed images
    """
    process_dict = Manager().dict()
    process = Process(target=compress, args=[frames, model, process_dict, print_log_messages, print_progress])

    process.start()
    process.join()

    return process_dict['return']
