# tab-cnn

### Guitar Tablature Estimation with a Convolutional Neural Network

###### This code supports the thesis "Automatic Tablature Estimation with Convolutional Neural Networks: Approaches and Limitations" and is a modified version of the code for the paper "Guitar Tablature Estimation with a Convolutional Neural Network" presented at the 20th Conference of the International Society for Music Information Retrieval (ISMIR 2019).

###### For basic instructions on how to run the model, visit https://www.github.com/andywiggins/tab-cnn.

# Additions to the readme by me:
 
After downloading GuitarSet as described in Step 1, run `tab-cnn/data/DataAugmentation.py` to perform the data augmentation, optionally providing system arguments to define the range of the transposition. This will produce transposed .wav and .jams files in the GuitarSet folder. Then, run the preprocessing and use the `output_id.csv` file during training to use the augmented data set.

The file `tab-cnn/data/ModelVariants.py` provides the model architecture for variations on the original tab-cnn model described in the article ([link]). These can be loaded in the training script `tab-cnn/model/model.py`.

Before training the model, verify the settings used for training in `tab-cnn/model/TabCNN.py`: 
- choosing the correct `id.csv` file (such as when using the file produced during augmentation)
- correctly configuring the experiment name, which determines the model variant chosen from `tab-cnn/model/ModelVariants.py
- correctly setting the EVALUATE parameter (which defines whether to evaluate the model between folds) and the LOAD_MODEL parameter (used to load and evaluate a pre-trained model from `tab-cnn/model/saved`
- correctly setting MODEL_PATH if using LOAD_MODEL.








