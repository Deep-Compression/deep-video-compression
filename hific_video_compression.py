'''
    This script can be used to compress and decompress video files. The approach is based on the hific model. Actually,
    to compress a video file, the video frames are separated and each frame will be compress using the hific model. All
    the compress frames are exported into a bitstring and written to file. This file represents the compressed video. To
    decompress a file, each frame is decompressed separately and the frames are fused into the final video file.

    @author: Felix Beutter
'''
import os
import pickle
import sys

import cv2
import numpy as np
import torch

from sepconv_slomo.run import estimate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from tf_compression.models.tfci import import_metagraph, instantiate_signature

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


def compress(input_file, output_file='compressed_video.dvc', model='hific-lo', interpolation='linear',
             num_intermediate_frames=1):
    """
        Compresses a video file using the hific generative image compression.

        :param input_file: Video to compress
        :param output_file: File name of compressed video
        :param model: hific model
        :param interpolation: Method to interpolate frames for further data reduction (None for no interpolation)
        :param num_intermediate_frames: Number of frames to interpolate between to compressed ones
    """
    video_capture = cv2.VideoCapture(input_file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < 1:
        print('ERROR: Video has no frames, compression will not be performed')
        exit(0)

    print("Starting compression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')

    success, cv_image = video_capture.read()
    n = 0

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    num_end_frames = (num_frames - 1) % (num_intermediate_frames + 1) + 1

    dictionary = {'fps': fps, 'model': model, 'num_frames': num_frames, 'interpolation': interpolation,
                  'num_intermediate_frames': num_intermediate_frames, 'num_end_frames': num_end_frames}

    with tf.Graph().as_default():
        signature_defs = import_metagraph(model)
        inputs, outputs = instantiate_signature(signature_defs['sender'])

        inputs = inputs['input_image']
        outputs = [outputs[k] for k in sorted(outputs) if k.startswith('channel:')]

        print_progress_bar(0, num_frames)

        with tf.Session() as sess:
            for _ in range(num_frames):
                if interpolation is None or n % (num_intermediate_frames + 1) == 0 or n >= num_frames - num_end_frames:
                    np_image = np.asarray(cv_image)
                    np_image = np.expand_dims(np_image, 0)

                    arrays = sess.run(outputs, feed_dict={inputs: np_image})

                    packed = tfc.PackedTensors()
                    packed.pack(outputs, arrays)

                    dictionary[str(n)] = packed.string

                n += 1
                print_progress_bar(n, num_frames)

                _, cv_image = video_capture.read()

    pickle.dump(dictionary, open(output_file, 'wb'))


def decompress(input_file, output_file='decompressed_video.mp4'):
    """
        Decompresses a compressed video file using the hific generative image compression.

        :param input_file: Compressed video to decompress
        :param output_file: File name of decompressed video
    """
    dictionary = pickle.load(open(input_file, 'rb'))

    num_frames = dictionary['num_frames']
    num_intermediate_frames = dictionary['num_intermediate_frames']
    num_end_frames = dictionary['num_end_frames']
    interpolation = dictionary['interpolation']

    print("Starting decompression of file '" + input_file + "' (" + str(num_frames) + ' frames)...')
    frames = [None] * num_frames
    n = 0

    with tf.Graph().as_default():
        signature_defs = import_metagraph(dictionary['model'])
        inputs, outputs = instantiate_signature(signature_defs['receiver'])

        inputs = [inputs[k] for k in sorted(inputs) if k.startswith('channel:')]

        progress = 0
        print_progress_bar(progress, num_frames)

        with tf.Session() as sess:
            for _ in range(num_frames):
                if interpolation is None or n % (num_intermediate_frames + 1) == 0 or n >= num_frames - num_end_frames:
                    bitstring = dictionary[str(n)]
                    arrays = tfc.PackedTensors(bitstring).unpack(inputs)

                    image = sess.run(outputs['output_image'], feed_dict=dict(zip(inputs, arrays)))

                    image = np.asarray(image)
                    image = np.squeeze(image, 0)
                    image = np.round(image)

                    frames[n] = image.astype(np.uint8)

                    progress += 1

                n += 1
                print_progress_bar(progress, num_frames)

        if interpolation is not None:
            for n in range(num_frames - num_end_frames):
                if n % (num_intermediate_frames + 1) == 0:
                    # frames[n + 1:n + 1 + num_intermediate_frames] = linear_interpolation(frames[n], frames[
                    #    n + 1 + num_intermediate_frames], num_intermediate_frames)

                    frames[n + 1] = sepconv_slomo_interpolation(frames[n], frames[n + 2])

                    progress += num_intermediate_frames
                    print_progress_bar(progress, num_frames)

    frame_shape = np.shape(image)
    width, height = frame_shape[1], frame_shape[0]

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), dictionary['fps'], (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


def linear_interpolation(first_frame, last_frame, num_intermediate_frames):
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
        :return: Numpy array of generated intermediate frames
    """
    first_image_data = np.array(first_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    first_tensor = torch.FloatTensor(np.ascontiguousarray(first_image_data))

    second_image_data = np.array(second_frame)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    second_tensor = torch.FloatTensor(np.ascontiguousarray(second_image_data))

    output = estimate(first_tensor, second_tensor)

    return (output.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)


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
