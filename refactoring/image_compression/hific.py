"""
    Created on 18 Feb 2021 09:19am

    @author Felix Beutter
"""
import os
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from ..helper import print_progress_bar


def _import_metagraph(model='hific-lo'):
    """
        Imports HiFiC model metagraph.

        :param model: HiFiC model ('hific-lo', 'hific-mi' or 'hific-hi')
        :return: Signature definition of imported metagraph
    """
    model_path = os.path.join('..resources', model + '.metagraph')

    try:
        with tf.io.gfile.GFile(model_path, "rb") as file:
            string = file.read()

    except tf.errors.NotFoundError:
        raise RuntimeError('Missing hific model metagraph file (\'{}\')'.format(model))

    metagraph = tf.MetaGraphDef()
    metagraph.ParseFromString(string)

    tf.train.import_meta_graph(metagraph)

    with tf.Graph().as_default():
        tf.import_graph_def(metagraph.graph_def)

    return metagraph.signature_def


def _instantiate_signature(signature_def):
    """
        Instantiates signature of metagraph.

        :param signature_def: Signature definition
        :return: Inputs and outputs
    """
    graph = tf.get_default_graph()

    inputs = {
        k: graph.get_tensor_by_name(v.name)
        for k, v in signature_def.inputs.items()
    }

    outputs = {
        k: graph.get_tensor_by_name(v.name)
        for k, v in signature_def.outputs.items()
    }

    return inputs, outputs


def compress_frames_hific(frames, model='hific-lo'):
    """
        Compresses a list of frames using a trained hific model.

        :param frames: Frames to compress (numpy arrays)
        :param model: Hific model ('hific-lo', 'hific-mi' or 'hific-hi')
        :return: Packed tensors of the compressed images
    """
    print('Starting compression of frames using HiFiC ({} frames)...'.format(len(frames)))

    packed_tensors = []
    print_progress_bar(0, len(frames))

    with tf.Graph().as_default():
        if model not in ['hific-lo', 'hific-mi', 'hific-hi']:
            raise RuntimeError('Invalid hific model name \'' + model + '\'!')

        signature_def = _import_metagraph(model)
        inputs, outputs = _instantiate_signature(signature_def['sender'])

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


def decompress_tensors_hific(packed_tensors, model):
    """
        Decompresses a list of packed tensors of images compressed with a hific network.

        :param packed_tensors: Packed tensors
        :param model: Hific model used for image compression
        :return: Decompressed frames (as numpy arrays)
    """
    print('Decompressing packed tensors using hific...')

    frames = []
    print_progress_bar(0, len(packed_tensors))

    with tf.Graph().as_default():
        signature_def = _import_metagraph(model)
        inputs, outputs = _instantiate_signature(signature_def['receiver'])

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

    return frames
