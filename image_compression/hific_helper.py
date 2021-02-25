"""
    Created on 18 Feb 2021 09:19am

    @author Felix Beutter
"""
import os
import tensorflow.compat.v1 as tf


def import_metagraph(model='hific-lo'):
    """
        Imports HiFiC model metagraph.

        :param model: HiFiC model ('hific-lo', 'hific-mi' or 'hific-hi')
        :return: Signature definition of imported metagraph
    """
    model_path = os.path.join('resources', model + '.metagraph')

    try:
        with tf.io.gfile.GFile(model_path, "rb") as file:
            string = file.read()

    except tf.errors.NotFoundError:
        raise RuntimeError('Missing hific model metagraph file (\'{}\')'.format(model_path))

    metagraph = tf.MetaGraphDef()
    metagraph.ParseFromString(string)

    tf.train.import_meta_graph(metagraph)

    with tf.Graph().as_default():
        tf.import_graph_def(metagraph.graph_def)

    return metagraph.signature_def


def instantiate_signature(signature_def):
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
