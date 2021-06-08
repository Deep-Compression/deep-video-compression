import os
from pathlib import Path

from PIL import Image

from config import *
from helper.print_progress_bar import print_progress_bar


def get_closest_quality(image, target_size):  # dc_file):
    # target_size = os.path.getsize(dc_file)

    min_quality, max_quality = 1, 95

    if get_jpeg_size(image, min_quality) > target_size:
        return min_quality

    if get_jpeg_size(image, max_quality) < target_size:
        return max_quality

    while min_quality != max_quality:
        quality = int((min_quality + max_quality) / 2)
        size = get_jpeg_size(image, quality)

        if size < target_size:
            if min_quality + 1 == max_quality:
                return max_quality

            min_quality = quality

        elif size > target_size:
            max_quality = quality

        else:
            return quality

    return min_quality


def get_jpeg_size(image, quality):
    image.save('tmp.jpg', optimizer=True, quality=quality)
    size = os.path.getsize('tmp.jpg')
    os.remove('tmp.jpg')

    return size


num_sequences = 0

for _, _, files in os.walk(DATASET_DIR):
    if 'im1.png' in files:
        num_sequences += 1

for model in MODELS:
    for depth in INTERPOLATION_DEPTHS:
        if depth == 1:
            dc_indices = [1, 3, 5]
        else:
            dc_indices = [1]

        print('JPEG Compression of dataset files with file sizes similar to model ' + model + ' and depth {}...'.format(
            depth))

        n = 0
        print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

        for root, _, files in os.walk(DATASET_DIR):
            if 'im1.png' in files:
                dc_root = COMPRESSED_DIR + '/' + model + root[10:]
                dc_files_size = 0

                for dc_index in dc_indices:
                    dc_files_size += os.path.getsize(dc_root + '/im' + str(dc_index) + '.dc')

                if depth == 1:
                    bytes_per_image = dc_files_size / 6
                else:
                    bytes_per_image = dc_files_size / 4

                out_root = JPEG_COMPRESSED_DIR + '/' + model + '/depth_' + str(depth) + root[10:]

                s = 6 if depth == 1 else 4

                for i in range(s):
                    image = Image.open(root + '/im' + str(i + 1) + '.png')

                    quality = get_closest_quality(image, bytes_per_image)

                    Path(out_root).mkdir(parents=True, exist_ok=True)
                    image.save(out_root + '/im' + str(i + 1) + '.jpg', optimizer=True, quality=quality)

                n += 1
                print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

'''
import os
from pathlib import Path

from PIL import Image

from config import *
from helper.print_progress_bar import print_progress_bar


def get_closest_quality(image, dc_file):
    target_size = os.path.getsize(dc_file)

    min_quality, max_quality = 1, 95

    if get_jpeg_size(image, min_quality) > target_size:
        return min_quality

    if get_jpeg_size(image, max_quality) < target_size:
        return max_quality

    while min_quality != max_quality:
        quality = int((min_quality + max_quality) / 2)
        size = get_jpeg_size(image, quality)

        if size < target_size:
            if min_quality + 1 == max_quality:
                return max_quality

            min_quality = quality

        elif size > target_size:
            max_quality = quality

        else:
            return quality

    return min_quality


def get_jpeg_size(image, quality):
    image.save('tmp.jpg', optimizer=True, quality=quality)
    size = os.path.getsize('tmp.jpg')
    os.remove('tmp.jpg')

    return size


num_sequences = 0

for _, _, files in os.walk(DATASET_DIR):
    if 'im1.png' in files:
        num_sequences += 1

for model in MODELS:
    print('JPEG Compression of dataset files with file sizes similar to model ' + model + '...')

    n = 0
    print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

    for root, _, files in os.walk(DATASET_DIR):
        if 'im1.png' in files:
            out_root = JPEG_COMPRESSED_DIR + '/' + model + root[10:]
            files = [file for file in files if len(file) == 7]

            for file in files:
                image = Image.open(root + '/' + file)
                dc_file = COMPRESSED_DIR + '/' + model + root[10:] + '/' + file[:-3] + 'dc'

                quality = get_closest_quality(image, dc_file)

                Path(out_root).mkdir(parents=True, exist_ok=True)
                image.save(out_root + '/' + file[:-3] + 'jpg', optimizer=True, quality=quality)

            n += 1
            print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))
'''
