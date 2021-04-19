import keras
import os
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Activation, concatenate, LSTM, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, Lambda, TimeDistributed
from keras import backend as K

NUM_CLASSES = 21
NUM_STRINGS = 6
INPUT_SHAPE = (192, 9, 1)
CHORD_INPUT_SHAPE = (48,)

def softmax_by_string(t):
    sh = K.shape(t)
    string_sm = []
    for i in range(NUM_STRINGS):
        string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
    return K.concatenate(string_sm, axis=1)


def catcross_by_string(target, output):
    loss = 0
    for i in range(NUM_STRINGS):
        loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
    return loss


def avg_acc(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))


def build_model(experiment='base'):

    print('Running experiment "{}"'.format(experiment))

    model = Sequential()

    if any([s in experiment for s in ['base', 'augmented', 'subset']]):

        print('just loaded exp_base')

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))  # no activation

    if 'adapt_02' in experiment:

        model.add(Conv2D(32, kernel_size=(9, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 9)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    if 'adapt_03' in experiment:

        model.add(Conv2D(32, kernel_size=(9, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    if 'adapt_04' in experiment:

        model.add(Conv2D(32, kernel_size=(27, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    if 'adapt_06' in experiment:

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (50, 1), activation='relu'))
        model.add(Conv2D(64, (27, 1), activation='relu'))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    if 'adapt_07' in experiment:

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (12, 3), padding='valid', activation='relu'))
        model.add(Conv2D(64, (23, 1), padding='valid', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(NUM_CLASSES * NUM_STRINGS, (1, 1), padding='valid', activation='linear'))
        model.add(AveragePooling2D(3, 13))


    if 'adapt_08' in experiment:

        model.add(Conv2D(32, kernel_size=(27, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))

    if 'adapt_09' in experiment:

        model.add(Conv2D(32, kernel_size=(15, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (9, 1), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    if 'adapt_10' in experiment:

        model.add(Conv2D(32, kernel_size=(27, 1),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (9, 9), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES * NUM_STRINGS))


    model.add(Reshape((NUM_STRINGS, NUM_CLASSES)))
    model.add(Activation(softmax_by_string))

    model.compile(loss=catcross_by_string,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=[avg_acc])


    if 'chords_01' in experiment:

        # requires include_chords=True on Data Generator
        # requires path setup to spec_repr_new

        print('Running chord input model experiment')

        input_1 = Input(shape=INPUT_SHAPE)
        convnet = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_1)
        convnet = Conv2D(64, (3, 3), activation='relu')(convnet)
        convnet = Conv2D(64, (3, 3), activation='relu')(convnet)
        convnet = MaxPooling2D(pool_size=(2, 3))(convnet)
        convnet = Dropout(0.25)(convnet)
        convnet = Flatten()(convnet)

        chord_input = Input(shape=CHORD_INPUT_SHAPE)

        merged = concatenate([convnet, chord_input])

        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(NUM_CLASSES * NUM_STRINGS)(merged)
        merged = Reshape((NUM_STRINGS, NUM_CLASSES))(merged)
        merged = Activation(softmax_by_string)(merged)

        model = Model(inputs=[input_1, chord_input], outputs=merged)
        model.compile(loss=catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[avg_acc])


    if 'chords_02' in experiment:

        # requires include_chords=True on Data Generator
        # requires path setup to spec_repr_new

        print('Running chord input model experiment')

        input_1 = Input(shape=INPUT_SHAPE)
        convnet = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_1)
        convnet = Conv2D(64, (3, 3), activation='relu')(convnet)
        convnet = Conv2D(64, (3, 3), activation='relu')(convnet)
        convnet = MaxPooling2D(pool_size=(2, 3))(convnet)
        convnet = Dropout(0.25)(convnet)
        convnet = Flatten()(convnet)
        convnet = Dense(128, activation='relu')(convnet)
        convnet = Dropout(0.5)(convnet)

        chord_input = Input(shape=CHORD_INPUT_SHAPE)
        merged = concatenate([convnet, chord_input])

        merged = Dense(NUM_CLASSES * NUM_STRINGS)(merged)
        merged = Reshape((NUM_STRINGS, NUM_CLASSES))(merged)
        merged = Activation(softmax_by_string)(merged)

        model = Model(inputs=[input_1, chord_input], outputs=merged)
        model.compile(loss=catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[avg_acc])


    if 'split_path_01' in experiment:

        input_layer = Input(shape=INPUT_SHAPE)

        model_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
        model_1 = Conv2D(64, (3, 3), activation='relu')(model_1)
        model_1 = Conv2D(64, (3, 3), activation='relu')(model_1)
        model_1 = MaxPooling2D(pool_size=(2, 2))(model_1)
        model_1 = Dropout(0.25)(model_1)
        model_1 = Flatten()(model_1)

        model_2 = Conv2D(32, kernel_size=(27, 1), activation='relu')(input_layer)
        model_2 = Conv2D(64, kernel_size=(9, 1), activation='relu')(model_2)
        model_2 = MaxPooling2D(pool_size=(2, 9))(model_2)
        model_2 = Dropout(0.25)(model_2)
        model_2 = Flatten()(model_2)

        merged = concatenate([model_1, model_2])

        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(NUM_CLASSES * NUM_STRINGS)(merged)
        merged = Reshape((NUM_STRINGS, NUM_CLASSES))(merged)
        merged = Activation(softmax_by_string)(merged)

        model = Model(inputs=[input_layer], outputs=merged)
        model.compile(loss=catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[avg_acc])


    if 'LSTM_01' in experiment:

        input_shape = (None, 192, 9, 1)

        input_layer = Input(shape=input_shape)

        print(input_layer.shape)
        cnn = TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'))(input_layer)
        cnn = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu'))(cnn)
        cnn = TimeDistributed(Conv2D(64, kernel_size=(3, 3), activation='relu'))(cnn)
        cnn = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(cnn)
        cnn = Dropout(0.25)(cnn)
        cnn = TimeDistributed(Flatten())(cnn)
        cnn = TimeDistributed(Dense(128, activation='relu'))(cnn)
        cnn = Dropout(0.5)(cnn)

        lstm = LSTM(256, return_sequences=True)(cnn)
        # could try blstm and concat here

        print(lstm.shape)

        fc = TimeDistributed(Dense(NUM_CLASSES * NUM_STRINGS))(lstm)
        fc = TimeDistributed(Reshape((NUM_STRINGS, NUM_CLASSES)))(fc)
        fc = TimeDistributed(Activation(softmax_by_string))(fc)

        print(fc.shape)

        model = Model(inputs=[input_layer], outputs=fc)

        model.compile(loss=catcross_by_string,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=[avg_acc])

    return model