'''
    This script can be used to compress and decompress video files. The approach is based on the hific model. Actually,
    to compress a video file, the video frames are separated and each frame will be compress using the hific model. All
    the compress frames are exported into a bitstring and written to file. This file represents the compressed video. To
    decompress a file, each frame is decompressed separately and the frames are fused into the final video file.
'''
import os
import sys
import cv2
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from compression.models.tfci import import_metagraph, instantiate_signature

USAGE_MESSAGE = '''
Usage:
    python hific_video_compression.py <command> <file>
    
Commands:
    compress    Compresses a video file
    decompress  Decompresses a video file
'''

MODEL = 'hific-lo'


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', end='\r'):
    '''
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. '\r', '\r\n') (Str)
    '''
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=end)

    if iteration == total:
        print()


def compress(input_file, output_file):
    """
        Compresses a video file using the hific generative image compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Starting compression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')

    success, cv_image = video_capture.read()
    n = 0
    dictionary = {}

    with tf.Graph().as_default():
        signature_defs = import_metagraph(MODEL)
        inputs, outputs = instantiate_signature(signature_defs['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        print_progress_bar(0, num_frames, prefix='Progress:', suffix='Complete', length=50)

        with tf.Session() as sess:
            while success:
                np_image = np.asarray(cv_image)
                np_image = np.expand_dims(np_image, 0)

                arrays = sess.run(outputs, feed_dict={inputs: np_image})

                packed = tfc.PackedTensors()
                packed.pack(outputs, arrays)

                dictionary[str(n)] = packed.string

                n += 1
                print_progress_bar(n, num_frames, prefix='Progress:', suffix='Complete', length=50)

                success, cv_image = video_capture.read()

    pickle.dump(dictionary, open(output_file, 'wb'))


def decompress(input_file, output_file):
    """
        Decompresses a compressed video file using the hific generative image compression.

        :param input_file: Compressed video to decompress
        :param output_file: File name of decompressed video
    """
    dictionary = pickle.load(open(input_file, 'rb'))
    num_frames = len(dictionary.keys())

    print("Starting decompression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')
    frames = []

    with tf.Graph().as_default():
        signature_defs = import_metagraph(MODEL)
        inputs, outputs = instantiate_signature(signature_defs['receiver'])

        inputs = [inputs[k] for k in sorted(inputs) if k.startswith('channel:')]

        print_progress_bar(0, num_frames, prefix='Progress:', suffix='Complete', length=50)

        with tf.Session() as sess:
            for n, key in enumerate(dictionary.keys()):
                bitstring = dictionary[key]
                arrays = tfc.PackedTensors(bitstring).unpack(inputs)

                image = sess.run(outputs['output_image'], feed_dict=dict(zip(inputs, arrays)))

                image = np.asarray(image)
                image = np.squeeze(image, 0)
                image = np.round(image)

                frames.append(image.astype(np.uint8))
                print_progress_bar(n + 1, num_frames, prefix='Progress:', suffix='Complete', length=50)

    frame_shape = np.shape(image)
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


def main():
    error = False

    if len(sys.argv) != 4:
        print('ERROR: syntax error')
        error = True

    elif sys.argv[1] not in ['compress', 'decompress']:
        print('ERROR: invalid command (' + sys.argv[1] + ')')
        error = True

    if error:
        print(USAGE_MESSAGE)
        exit(1)

    if sys.argv[1] == 'compress':
        compress(sys.argv[2], sys.argv[3])

    else:
        decompress(sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    main()
