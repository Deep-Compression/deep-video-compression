import os

from config import *

for model in MODELS:
    total_file_size = 0
    num_files = 0

    for root, _, files in os.walk('output/compressed/' + model):
        for file in files:
            if file.endswith('.dc'):
                total_file_size += os.path.getsize(root + '/' + file)
                num_files += 1

    print(model, total_file_size / num_files)

"""
    BYTES PER IMAGE -> BITRATE (BITS / SECOND, 25 FPS)

    interpolation depth: 1
    
        hific-hi: 2783.0741400087640 -> 556614.82800175280
        hific-mi: 1986.9772677475898 -> 397395.45354951796
        hific-lo: 1107.8140337423313 -> 221562.80674846625

    interpolation depth: 2
    
        hific-hi: 1391.537070004382 -> 278307.41400087640
        hific-mi: 993.4886338737949 -> 198697.72677475898
        hific-lo: 553.9070168711656 -> 110781.40337423312
"""



