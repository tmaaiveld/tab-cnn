import os
import numpy as np
import jams
from scipy.io import wavfile
import sys
import librosa
from keras.utils import to_categorical

class TabDataReprGen:
    
    def __init__(self, mode="c"):
        # file path to the GuitarSet dataset
        path = "GuitarSet/"
        self.path_audio = path + "audio/audio_mic/"
        self.path_anno = path + "annotation/"

        # labeling parameters
        self.string_midi_pitches = [40,45,50,55,59,64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2 # for open/closed

        self.notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.chord_types = ['maj', 'min', '7', 'hdim7']
        self.chords = ['{}:{}'.format(note, chord_type) for note in self.notes
                                                        for chord_type in self.chord_types]
        
        # prepresentation and its labels storage
        self.output = {}

        # out-of-bounds marker storage
        self.oob = []
        
        # preprocessing modes
        #
        # c = cqt
        # m = melspec
        # cm = cqt + melspec
        # s = stft
        #
        self.preproc_mode = mode
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        
        # CQT parameters
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        
        # STFT parameters
        self.n_fft = 2048
        self.hop_length = 512
        
        # save file path
        self.save_path = "spec_repr_new/" + self.preproc_mode + "/"

        # ID file path
        self.path_id = "spec_repr/output_id.csv"


    def load_rep_and_labels_from_raw_file(self, filename):
        file_audio = self.path_audio + filename + "_mic.wav"
        file_anno = self.path_anno + filename + ".jams"

        print(file_anno)

        jam = jams.load(file_anno)
        self.sr_original, data = wavfile.read(file_audio)
        self.sr_curr = self.sr_original
        
        # preprocess audio, store in output dict
        self.output["repr"] = np.swapaxes(self.preprocess_audio(data),0,1)
        
        # construct labels
        frame_indices = range(len(self.output["repr"]))
        times = librosa.frames_to_time(frame_indices, sr=self.sr_curr, hop_length=self.hop_length)
        
        # loop over all strings and sample annotations
        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(round(string_label_samples[i][0]) - self.string_midi_pitches[string_num])

            labels.append([string_label_samples])

            # print(labels[0])
            # if len(labels) in range(1310, 1320):
            #     print(string_label_samples)


        # quit()

        labels = np.array(labels)
        # remove the extra dimension 
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels,0,1)

        # clean labels
        labels = self.clean_labels(labels)

        # store labels
        self.output["labels"] = labels

        # create chord annotation
        chord_anno = jam.annotations['chord'][0]
        chord_samples = chord_anno.to_samples(times)

        # generate one hot coded entries and store
        self.output["chord"] = self.map_chords(chord_samples)

        # write IDs to ID file
        self.write_ids(filename)

        return len(labels)

    def map_chords(self, samples):

        result = []
        for sample in samples:
            chord_label = np.zeros(len(self.chords))
            chord_label[self.chords.index(sample[0])] = 1

            result.append(chord_label)

        return result

    def write_ids(self, filename, id_csv_path="spec_repr/output_id.csv"):

        n_frames = len(self.output["repr"])

        frame_idx = np.nonzero(np.amin(np.array(self.oob).reshape(n_frames, -1), axis=1))[0]

        frame_names = [filename + '_' + str(i) for i in frame_idx]

        with open(id_csv_path, 'a') as f:
            [f.write(name + '\n') for name in frame_names]

    def correct_numbering(self, n):
        n += 1

        if n < 0:
            n = 0
            self.oob.append(1) # include

        elif n > self.highest_fret + 1:
            n = 0
            self.oob.append(0) # omit

        else:
            self.oob.append(1) # include

        return n
    
    def categorical(self, label):
        return to_categorical(label, self.num_classes)
    
    def clean_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)
    
    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])

    def preprocess_audio(self, data):
        data = data.astype(float)
        data = np.asfortranarray(data) # added for support on Windows systems

        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(data, self.sr_original, self.sr_downs)
            self.sr_curr = self.sr_downs
        if self.preproc_mode == "c":
            data = np.abs(librosa.cqt(data,
                hop_length=self.hop_length, 
                sr=self.sr_curr, 
                n_bins=self.cqt_n_bins, 
                bins_per_octave=self.cqt_bins_per_octave))
        elif self.preproc_mode == "m":
            data = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
        elif self.preproc_mode == "cm":
            cqt = np.abs(librosa.cqt(data, 
                hop_length=self.hop_length, 
                sr=self.sr_curr, 
                n_bins=self.cqt_n_bins, 
                bins_per_octave=self.cqt_bins_per_octave))
            mel = librosa.feature.melspectrogram(y=data, sr=self.sr_curr, n_fft=self.n_fft, hop_length=self.hop_length)
            data = np.concatenate((cqt,mel),axis = 0)
        elif self.preproc_mode == "s":
            data = np.abs(librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length))
        else:
            print "invalid representation mode."

        return data

    def save_data(self, filename):
        np.savez(filename, **self.output)
        
    def get_nth_filename(self, n):
        # returns the filename with no extension
        filenames = np.sort(np.array(os.listdir(self.path_anno)))
        filenames = filter(lambda x: x[-5:] == ".jams", filenames)
        return filenames[n][:-5] 
    
    def load_and_save_repr_nth_file(self, n):
        # filename has no extension
        filename = self.get_nth_filename(n)
        num_frames = self.load_rep_and_labels_from_raw_file(filename)
        print "done: " + filename + ", " + str(num_frames) + " frames"
        return
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_data(save_path + filename + ".npz")
        
def main(args):

    n = args[0]
    m = args[1]

    gen = TabDataReprGen(mode=m)
    gen.load_and_save_repr_nth_file(n)


if __name__ == "__main__":
    main(args)
