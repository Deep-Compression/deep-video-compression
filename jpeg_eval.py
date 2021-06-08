import os

import cv2
import lpips
import numpy as np

from config import *
from helper.print_progress_bar import print_progress_bar
from metrics.PerceptualSimilarity.run_lpips import calculate_lpips
from metrics.multi_scale_ssim import multi_scale_ssim
from metrics.psnr import calculate_psnr

num_sequences = 1000

# for _, _, files in os.walk(DATASET_DIR):
#     if 'im1.png' in files:
#         num_sequences += 1

lpips_model = lpips.LPIPS(verbose=False)

for model in MODELS:
    for depth in INTERPOLATION_DEPTHS:
        print('Eval of jpeg with bit rate similar to ' + model + ' and depth {}...'.format(depth))

        n = 0
        print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

        msssim, psnr, lpips = 0, 0, 0

        for root, _, files in os.walk('output/jpeg/' + model + '/depth_' + str(depth)):
            if 'im1.jpg' in files:
                frames, original_frames = [], []
                original_root = DATASET_DIR + '/' + '/'.join(root.split('/')[-2:])

                for file in files:
                    frames.append(cv2.imread(root + '/' + file))
                    original_frames.append(cv2.imread(original_root + '/' + file[:-3] + 'png'))

                frames = np.array(frames)
                original_frames = np.array(original_frames)

                for metric in ['lpips']:
                    if metric == 'msssim':
                        msssim += multi_scale_ssim(original_frames, frames)

                    elif metric == 'psnr':
                        for i in range(len(frames)):
                            psnr += calculate_psnr(original_frames[i], frames[i]) / len(frames)

                    elif metric == 'lpips':
                        for i in range(len(frames)):
                            lpips += np.squeeze(
                                calculate_lpips(original_frames[i], frames[i], lpips_model).detach().numpy()) / len(
                                frames)

                    else:
                        raise Exception('Invalid evaluation metric \'{}\''.format(metric))

                n += 1
                print_progress_bar(n, num_sequences, suffix='({}/{} sequences)'.format(n, num_sequences))

                if n == num_sequences:
                    break

        mean_msssim = msssim / num_sequences
        mean_psnr = psnr / num_sequences
        mean_lpips = lpips / num_sequences

        string = '\nEvaluation of: {} bitrate similar to {} depth_{}\n'.format('output/jpeg', model, depth)
        string += 'Mean MS-SSIM: {}\n'.format(mean_msssim)
        string += 'Mean PSNR: {}\n'.format(mean_psnr)
        string += 'Mean LPIPS: {}\n'.format(mean_lpips)

        print(string)

        with open('jpeg_evaluation_results.txt', 'a') as f:
            f.write(string)