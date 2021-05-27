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
    print('Compression of dataset files using ' + model + '...')

    n = 0
    print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

    with tf.Graph().as_default():
        if model not in ['hific-lo', 'hific-mi', 'hific-hi']:
            raise RuntimeError('Invalid hific model name \'' + model + '\'!')

        signature_def = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_def['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        with tf.Session() as session:
            total_original_size = 0
            total_compressed_size = 0

            for root, _, files in os.walk(DATASET_DIR):
                if 'im1.png' in files:
                    out_root = COMPRESSED_DIR + '/' + model + root[10:]
                    files = [file for file in files if len(file) == 7]

                    for file in files:
                        frame = cv2.imread(root + '/' + file)
                        frame = np.expand_dims(frame, 0)

                        arrays = session.run(outputs, feed_dict={inputs: frame})
                        packed = tfc.PackedTensors()
                        packed.pack(outputs, arrays)

                        out_path = out_root + '/' + file[:-4] + '.dc'
                        Path('/'.join(out_path.split('/')[0:-1])).mkdir(parents=True, exist_ok=True)
                        pickle.dump(packed.string, open(out_path, 'wb'))

                        total_original_size += os.path.getsize(root + '/' + file)
                        total_compressed_size += os.path.getsize(out_root + '/' + file[:-4] + '.dc')

                    n += 1
                    print_progress_bar(n, NUM_SEQUENCES, suffix='({}/{} sequences)'.format(n, NUM_SEQUENCES))

    out_string = '\nMean compression factor for ' + model + ': ' + str(total_original_size / total_compressed_size)
    out_string += '\nTotal size of original sequence: ' + str(total_original_size)
    out_string += '\nTotal size of compressed sequence: ' + str(total_compressed_size) + '\n'
    print(out_string)

    with open('compression_output.txt', 'a') as f:
        f.write(out_string)
