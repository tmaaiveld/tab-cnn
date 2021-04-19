import os
import sys
import glob
import jams
import librosa

SAMPLING_RATE = 44100
DATA_DIR = '.\\GuitarSet' if len(sys.argv) < 2 else sys.argv[1]
STEP_RANGE = range(1,12) if len(sys.argv) < 4 else range(sys.argv[2], sys.argv[3])

def transpose_audio(path, step, save_dir=None):

    if save_dir is None:
        save_dir = os.path.dirname(path)

    print('shifting {} by {} steps...'.format(os.path.basename(path), step))

    output_path = save_dir + '\\{}_{}_mic.wav'.format(os.path.basename(path)[:-8], step)

    y, sr = librosa.load(path, sr=SAMPLING_RATE)

    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=step, bins_per_octave=12)
    librosa.output.write_wav(output_path, y_shifted, sr=SAMPLING_RATE, norm=False)


def tranpose_annotation(path, step, save_dir=None):

    if save_dir is None:
        save_dir = os.path.dirname(path)

    print('transposing {} by {}'.format(os.path.basename(path), step))

    # load in the file
    with open(path, 'r') as f:
        annotation = jams.load(f)

        # for each string annotation
        for string in range(6):
            # load the midi data
            string_midi = annotation['annotations']['note_midi'][string]

            # generate pitch-shifted data
            string_df = string_midi.to_dataframe()
            string_df['value'] = string_df['value'] + step

            output = string_df.to_dict(orient='records')

            # delete the original data
            annotation['annotations']['note_midi'][string].pop_data()

            # replace it
            annotation['annotations']['note_midi'][string].append_records(output)

        output_path = save_dir + '\\{}_{}.jams'.format(os.path.basename(path)[:-5], step)

        with open(output_path, 'w') as output_f:
            annotation.save(output_f)
            print('Annotation saved to {}'.format(output_path))

def main():

    if not os.path.exists(DATA_DIR):
        raise IOError('{} directory does not exist.'.format(os.path.abspath(DATA_DIR)))

    audio_dir = DATA_DIR + '\\audio\\audio_mic\\'
    anno_dir = DATA_DIR + '\\annotation\\'

    audio_paths = list(glob.glob(audio_dir + '*.wav'))
    annotation_paths = list(glob.glob(anno_dir + '*.jams'))

    print(audio_paths)
    for audio_path, annotation_path in zip(audio_paths, annotation_paths):
        for step in STEP_RANGE:
            transpose_audio(audio_path, step)
            tranpose_annotation(annotation_path, step)


if __name__ == '__main__':
    main()
