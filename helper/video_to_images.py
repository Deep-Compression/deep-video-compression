import sys
import cv2
from pathlib import Path

from progress_bar import print_progress_bar


def video_to_images(file, directory):
    """
        Converts a video file into its frames and saves them into a directory.

        :param file: Video file
        :param directory: Where the frames are stored
    """
    video_capture = cv2.VideoCapture(file)
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    _, image = video_capture.read()

    Path(directory).mkdir(parents=True, exist_ok=True)
    print_progress_bar(0, num_frames)

    for i in range(num_frames):
        cv2.imwrite(directory + '/frame_{}.png'.format(i), image)
        success, image = video_capture.read()

        print_progress_bar(i + 1, num_frames)


def main():
    """
        Main method. Parses arguments and calls video_to_image method.

        Usage: python video_to_images.py [video file] [directory to write frames into]
    """
    if len(sys.argv) != 3:
        raise RuntimeError('ERROR! Invalid number of arguments.')

    video_to_images(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
