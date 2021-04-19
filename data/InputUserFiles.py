import os
from multiprocessing import Pool
from TabDataReprGen import main
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

N_FILES = 4320
ID_CSV_PATH = "spec_repr/output_id.csv" # creates IDs (necessary for augmented dataset)
enable_multiprocessing = False

index = range(N_FILES)
modes = ['c'] * N_FILES

if __name__ == '__main__':

    # make a csv file listing the frame IDs to be used during model training
    if os.path.exists(ID_CSV_PATH):
        assert raw_input('Are you sure you want to overwrite {}? [y/n]'.format(os.path.basename(ID_CSV_PATH))) == 'y'

    with open(ID_CSV_PATH, 'w'): pass

    if not enable_multiprocessing:
        for i in zip(index, modes):
            main(i)

    else:
        pool = Pool(11)
        results = pool.map(main, zip(index, modes))