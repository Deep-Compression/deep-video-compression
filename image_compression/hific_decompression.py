"""
    Created on 18 Feb 2021 11:07am

    @author Felix Beutter
"""
import numpy as np
from multiprocessing import Manager, Process

import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from image_compression.hific_helper import import_metagraph, instantiate_signature
from helper.print_progress_bar import print_progress_bar


def decompress(packed_tensors, model, process_dict=None, print_log_messages=True, print_progress=True):
    """
        Decompresses a list of packed tensors of images compressed with a hific network.

        :param packed_tensors: Packed tensors
        :param model: Hific model used for image compression
        :param process_dict: Process dictionary to store results if method is executed in isolated process
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Decompressed frames (as numpy arrays)
    """
    if print_log_messages:
        print('Decompressing packed tensors using HiFiC... ({} tensors)'.format(len(packed_tensors)))

    frames = []

    if print_progress:
        print_progress_bar(0, len(packed_tensors))

    with tf.Graph().as_default():
        signature_def = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_def['receiver'])

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

                if print_progress:
                    print_progress_bar(i + 1, len(packed_tensors))

    if process_dict is not None:
        process_dict['return'] = frames

    return frames


def decompress_process(packed_tensors, model, print_log_messages=True, print_progress=True):
    """
        Decompresses a list of packed tensors of images compressed with a hific network (executed in an isolated
        process).

        :param packed_tensors: Packed tensors
        :param model: Hific model used for image compression
        :param print_log_messages: If True, log message are printed to console
        :param print_progress: If True, a progress bar is printed to console
        :return: Decompressed frames (as numpy arrays)
    """
    process_dict = Manager().dict()
    process = Process(target=decompress, args=[packed_tensors, model, process_dict, print_log_messages, print_progress])

    process.start()
    process.join()

    return process_dict['return']
