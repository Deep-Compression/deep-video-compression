import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow debug output

import cv2
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from config import *
from helper.print_progress_bar import print_progress_bar
from image_compression.hific_helper import import_metagraph, instantiate_signature


for model in MODELS:
    print('Decompression of dataset files using ' + model + '...')

    n = 0
    print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

    with tf.Graph().as_default():
        if model not in ['hific-lo', 'hific-mi', 'hific-hi']:
            raise RuntimeError('Invalid hific model name \'' + model + '\'!')

        signature_def = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_def['receiver'])

        inputs = [inputs[k] for k in sorted(inputs) if k.startswith('channel:')]

        with tf.Session() as session:
            for root, _, files in os.walk(COMPRESSED_DIR + '/' + model):
                if 'im1.dc' in files:
                    out_root = DECOMPRESSED_DIR + root[19:]
                    files = [file for file in files if len(file) == 6]

                    for file in files:
                        packed_tensor = pickle.load(open(root + '/' + file, 'rb'))
                        arrays = tfc.PackedTensors(packed_tensor).unpack(inputs)

                        frame = session.run(outputs['output_image'], feed_dict=dict(zip(inputs, arrays)))
                        frame = np.squeeze(frame, 0)
                        frame = np.round(frame)
                        frame = np.clip(frame, 0, 255)
                        frame = frame.astype(np.uint8)

                        out_path = out_root + '/' + file[:-3] + '.png'
                        Path('/'.join(out_path.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(out_path, frame)

                    n += 1
                    print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

        print()
