'''
    This script can be used to compress and decompress video files. The approach is based on the hific model. Actually,
    to compress a video file, the video frames are separated and each frame will be compress using the hific model. All
    the compress frames are exported into a bitstring and written to file. This file represents the compressed video. To
    decompress a file, each frame is decompressed separately and the frames are fused into the final video file.

    @author: Felix Beutter
'''
import os
import sys
import cv2
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from tensorflow_compression.models.tfci import import_metagraph, instantiate_signature

USAGE_MESSAGE = '''
Usage:
    python hific_video_compression.py <command> <input_file> [output_file] [model]
    
Commands:
    compress    Compresses a video file
    decompress  Decompresses a video file
    
Examples:
    python hific_video_compression.py compress video.mp4 compressed_video.dvc hific-hi
    python hific_video_compression.py decompress compressed_video.dvc decompressed_video.mp4
'''


def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='█', end='\r'):
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


def compress(input_file, output_file='compressed_video.dvc', model='hific-lo'):
    """
        Compresses a video file using the hific generative image tensorflow_compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
        :param model: hific model
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Starting tensorflow_compression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')

    success, cv_image = video_capture.read()
    n = 0

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    dictionary = {'fps': fps, 'model': model}

    with tf.Graph().as_default():
        signature_defs = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_defs['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        print_progress_bar(0, num_frames)

        with tf.Session() as sess:
            while success:
                np_image = np.asarray(cv_image)
                np_image = np.expand_dims(np_image, 0)

                arrays = sess.run(outputs, feed_dict={inputs: np_image})

                packed = tfc.PackedTensors()
                packed.pack(outputs, arrays)

                dictionary[str(n)] = packed.string

                n += 1
                print_progress_bar(n, num_frames)

                success, cv_image = video_capture.read()

    pickle.dump(dictionary, open(output_file, 'wb'))


def decompress(input_file, output_file='decompressed_video.mp4'):
    """
        Decompresses a compressed video file using the hific generative image tensorflow_compression.

        :param input_file: Compressed video to decompress
        :param output_file: File name of decompressed video
    """
    dictionary = pickle.load(open(input_file, 'rb'))
    num_frames = len(dictionary.keys()) - 2

    print("Starting decompression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')
    frames = []

    with tf.Graph().as_default():
        signature_defs = import_metagraph(dictionary['model'])
        inputs, outputs = instantiate_signature(signature_defs['receiver'])

        inputs = [inputs[k] for k in sorted(inputs) if k.startswith('channel:')]

        print_progress_bar(0, num_frames)

        with tf.Session() as sess:
            for n, key in enumerate(dictionary.keys()):
                if key not in ['fps', 'model']:
                    bitstring = dictionary[key]
                    arrays = tfc.PackedTensors(bitstring).unpack(inputs)

                    image = sess.run(outputs['output_image'], feed_dict=dict(zip(inputs, arrays)))

                    image = np.asarray(image)
                    image = np.squeeze(image, 0)
                    image = np.round(image)

                    frames.append(image.astype(np.uint8))
                    print_progress_bar(n + 1, num_frames)

    frame_shape = np.shape(image)
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), dictionary['fps'], (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


def main():
    error = False

    if len(sys.argv) < 3:
        print('ERROR: syntax error')
        error = True

    elif sys.argv[1] not in ['compress', 'decompress']:
        print('ERROR: invalid command (' + sys.argv[1] + ')')
        error = True

    else:
        if sys.argv[1] == 'compress':
            if len(sys.argv) not in range(3, 6):
                print('ERROR: syntax error')
                error = True

            elif len(sys.argv) == 5:
                if sys.argv[4] not in ['hific-lo', 'hific-mi', 'hific-hi']:
                    print('ERROR: unknown model (' + sys.argv[4] + ')')

        else:
            if len(sys.argv) not in range(3, 5):
                print('ERROR: syntax error')
                error = True

    if error:
        print(USAGE_MESSAGE)
        exit(1)

    if sys.argv[1] == 'compress':
        compress(*sys.argv[2:])

    else:
        decompress(*sys.argv[2:])


if __name__ == '__main__':
    main()
