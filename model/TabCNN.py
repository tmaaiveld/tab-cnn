''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import keras
import os
import json
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from keras import backend as K
from DataGenerator import DataGenerator
from SequentialDataGenerator import SequentialDataGenerator
from ModelVariants import build_model
import pandas as pd
import numpy as np
import datetime
from Metrics import *
from time import time, gmtime, strftime
import pickle


class TabCNN:

    def __init__(self,
                 batch_size=128,
                 epochs=8,
                 con_win_size=9,
                 spec_repr="c",
                 data_path="../data/spec_repr/",
                 id_file=None,
                 save_path="saved/",
                 workers=12,
                 include_chords=False):

        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.workers = workers
        self.model = None
        self.include_chords = include_chords

        self.load_IDs()

        self.save_folder = self.save_path + self.spec_repr + " " + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S") + "/"

        self.save_folder = self.save_folder.replace(':', '')

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"

        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = ["g0", "g1", "g2", "g3", "g4", "g5", "mean", "std dev"]

        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def partition_data(self, data_split):
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []

        print('partitioning data')

        for ID in self.list_IDs:

            guitarist = int(ID.split("_")[0])
            if guitarist == data_split:
                self.partition["validation"].append(ID)
            else:
                self.partition["training"].append(ID)

        print("Training set size: ", len(self.partition["training"]))
        print("Validation set size: ", len(self.partition["validation"]))

        self.training_generator = DataGenerator(self.partition['training'],
                                                data_path=self.data_path,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                spec_repr=self.spec_repr,
                                                con_win_size=self.con_win_size,
                                                include_chords=self.include_chords)

        self.validation_generator = DataGenerator(self.partition['validation'],
                                                  data_path=self.data_path,
                                                  batch_size=len(self.partition['validation']),
                                                  shuffle=False,
                                                  spec_repr=self.spec_repr,
                                                  con_win_size=self.con_win_size,
                                                  include_chords=self.include_chords)

        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)

    def log_model(self, experiment):
        with open(self.log_file, 'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file))
            fh.write("\nexperiment: " + str(experiment) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def load_pretrained(self, model_path):
        self.model.load_weights(model_path)
        print('loaded weights from ' + model_path + '.')

    def train(self):

        os.mkdir(self.split_folder + '/checkpoints/')

        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=self.split_folder +  '/checkpoints/',
                                            monitor="accuracy",
                                            save_best_only=True,
                                            save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor="accuracy", patience=2)
        ]

        self.hist = self.model.fit_generator(
            generator=self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=8,
            callbacks=callbacks,
        )

        json_file = self.split_folder + 'history.json'
        with open(json_file, 'w') as fh:
            json.dump(self.hist.history, fh)

    def save_weights(self):
        self.model.save_weights(self.split_folder + "weights.h5")

    def test(self):
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test, verbose=0)

    def save_predictions(self):
        np.savez(self.split_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] = self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv")

    ##################################


########### EXPERIMENT ###########
##################################

# checklist
# - correct experiment name
# - correct npz file directory
# - correct range of folds
# - correct ID file
# - correct DataGenerator settings (include_chords=T/F, spec_repr path? (not needed, remove default)
# - correct batch size

EXPERIMENT = 'augmented_+3_eval'
INCLUDE_CHORDS = 'chords' in EXPERIMENT
DATA_PATH = '../data/spec_repr/'  # folder should be fine for all experiments
FOLD_RANGE = (0, 6)
ID_FILE = "id.csv"
N_WORKERS = 12

EVALUATE = True
LOAD_MODEL = True
MODEL_PATH = r"saved/{}/".format(
    "c 2021-02-09 235235 (augmented +3)"
)

if __name__ == '__main__':

    start_time = time()

    tabcnn = TabCNN(data_path=DATA_PATH,
                    id_file=ID_FILE,
                    workers=N_WORKERS,
                    include_chords=INCLUDE_CHORDS)

    tabcnn.model = build_model(experiment=EXPERIMENT)

    print("logging model...")
    tabcnn.log_model(EXPERIMENT)

    for fold in range(*FOLD_RANGE):

        print("\nfold " + str(fold))
        tabcnn.partition_data(fold)

        print("building model...")
        tabcnn.model = build_model(experiment=EXPERIMENT)

        if LOAD_MODEL:

            model_path = MODEL_PATH + str(fold) + "/weights.h5"
            print('loading pretrained model...')
            tabcnn.load_pretrained(model_path)

        else:
            print("training...")
            tabcnn.train()
            tabcnn.save_weights()

        if not EVALUATE:
            continue

        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()

    if EVALUATE:
        print("saving results...")
        tabcnn.save_results_csv()

    try:
        print("Total model training time: {}".format(strftime("%H:%M:%S", gmtime(time() - start_time))))
    except:
        print('Time print failed')
