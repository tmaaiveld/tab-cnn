# tab-cnn

### Guitar Tablature Estimation with a Convolutional Neural Network

###### This code supports the thesis "Automatic Tablature Estimation with Convolutional Neural Networks: Approaches and Limitations" and is a modified version of the code for the paper "Guitar Tablature Estimation with a Convolutional Neural Network" presented at the 20th Conference of the International Society for Music Information Retrieval (ISMIR 2019) (www.github.com/andywiggins/tab-cnn).

# Original readme documentation (Andy Wiggins & Youngmoo Kim):

To visualize the system performance, `tab-cnn/demos/` contains video demonstrations showing predicted and ground truth tablature synced with input audio. To run the code, follow the instructions below.

### 0. Requirements

This project was made to be run with Python 2.7. You should have the following libraries/packages installed:
* numpy
* scipy
* pandas
* jams
* librosa
* keras
* tensorflow

### 1. Download dataset

Download the GuitarSet audio and annotation data from [here](https://zenodo.org/record/1422265/files/GuitarSet_audio_and_annotation.zip?download=1 "GuitarSet download"). (Thanks again to the authors for creating this awesome dataset!)

Unzip and place the downloaded GuitarSet folder in `tab-cnn/data/` so that in `tab-cnn/data/GuitarSet/` you have the following two folders:
* `annotation/`
* `audio/`

The remaining instructions assume that you are in the `tab-cnn/` folder.

### 2. Preprocess audio

Run the following line to preprocess different spectral representations for the audio files: 

  `bash data/Bash_TabDataReprGen.sh`

This will save the preprocessed data as compressed numpy (.npz) files in the `data/spec_repr/` directory.

### 3. (Optional) Set spectral representation

The default spectral spresentation is the Constant-Q Transform (CQT). To change representations, an argument can be added when creating the TabCNN object. Specifically, on line 193 in `model/TabCNN.py`, a `spec_repr` parameter can be set to one of the following:
* `spec_repr = "c"`, for CQT
* `spec_repr = "m"`, for Mel-scaled spectrogram (Melspec)
* `spec_repr = "cm"`, for CQT + Melspec concatenation
* `spec_repr = "s"`, for Short-time Fourier Transform (STFT)

### 4. Train and test model

Run the following line to train and test the TabCNN model:

`python model/TabCNN.py`

A summary log and a csv results file will be saved in a time-stamped folder within the `model/saved/` directory. Additionally, a folder for each fold of data will be created, containing the individual model's weights and predictions. 


# Additions to the readme by me:
 
After downloading GuitarSet as described in Step 1, run `tab-cnn/data/DataAugmentation.py` to perform the data augmentation, optionally providing system arguments to define the range of the transposition. This will produce transposed .wav and .jams files in the GuitarSet folder. Then, run the preprocessing and use the `output_id.csv` file during training to use the augmented data set.

Before training the model, verify the settings used for training in `tab-cnn/model/TabCNN.py`: 
- choosing the correct `id.csv` file (such as when using the file produced during augmentation)
- correctly configuring the experiment name, which determines the model variant chosen from `tab-cnn/model/ModelVariants.py
- correctly setting the EVALUATE parameter (which defines whether to evaluate the model between folds) and the LOAD_MODEL parameter (used to load and evaluate a pre-trained model from `tab-cnn/model/saved`
- correctly setting MODEL_PATH if using LOAD_MODEL.








