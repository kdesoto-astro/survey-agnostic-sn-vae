# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input, GRU, TimeDistributed
from keras.layers import Dense, concatenate
from keras.layers import RepeatVector
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping
import datetime
import os
import logging

from survey_agnostic_sn_vae.preprocessing import prep_input
from survey_agnostic_sn_vae.custom_nn_layers.sim_loss import SimilarityLossLayer
from survey_agnostic_sn_vae.custom_nn_layers.kl_loss import SamplingLayer, AnnealingCallback
from survey_agnostic_sn_vae.custom_nn_layers.recon_loss import ReconstructionLoss

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

NEURON_N_DEFAULT = 100
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 1000


def make_model(LSTMN, encodingN, maxlen, nfilts):
    """
    Make RAENN model

    Parameters
    ----------
    LSTMN : int
        Number of neurons to use in first/last layers
    encodingN : int
        Number of neurons to use in encoding layer
    maxlen : int
        Maximum LC length
    nfilts : int
        Number of filters in LCs

    Returns
    -------
    model : keras.models.Model
        RAENN model to be trained
    callbacks_list : list
        List of keras callbacks
    input_1 : keras.layer
        Input layer of RAENN
    encoded : keras.layer
        RAENN encoding layer
    """
    input_1 = Input((None, nfilts*3+1))
    input_2 = Input((maxlen, 2))

    # make encoder and decoder models separately
    encoder = Sequential()
    encoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            ), 
            name="enc1"
        )
    )
    encoder.add(
        GRU(
            LSTMN,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=False,
            name="enc2",
            #dropout=0.5,
        )
    )

    
    # DECODER
    decoder = Sequential()
    decoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ), name="dec1"
        )
    )
    decoder.add(
        TimeDistributed(
            Dense(
                nfilts,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
            ), name="dec2"
        )
    )
    sampling = Sampling()
    annealing = AnnealingCallback(sampling.beta,"cyclical",n_epochs)
    callbacks_list.append(annealing)
    
    encoded_mean_layer = Dense(
        encodingN, activation='linear', name="mu",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
    )
    
    encoded_log_var_layer = Dense(
        encodingN, activation='linear', name="sigma",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
    )
    
    encoded = encoder(input_1)
    
    # add KL loss
    encoded_mean = encoded_mean_layer(encoded)
    encoded_log_var = encoded_log_var_layer(encoded)
    encoded_sample = sampling([encoded_mean, encoded_log_var], True)
    
    # This just outputs the same input, but adds a loss term
    #encoded = SimilarityLossLayer(encoded, input_1)

    repeater = RepeatVector(maxlen)(encoded_sample)
    merged = concatenate([repeater, input_2], axis=-1)
    decoded = decoder(merged)
    
    model = Model([input_1, input_2], decoded)

    new_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    
    rl = ReconstructionLoss(nfilts)
    model.compile(optimizer=new_optimizer, loss=rl)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    callbacks_list = [es]
    return model, callbacks_list, input_1, encoded


def fit_model(model, callbacks_list, sequence, outseq, n_epoch):
    """
    Make RAENN model

    Parameters
    ----------
    model : keras.models.Model
        RAENN model to be trained
    callbacks_list : list
        List of keras callbacks
    sequence : numpy.ndarray
        Array LC flux times, values and errors
    outseq : numpy.ndarray
        An array of LC flux values and limiting magnitudes
    n_epoch : int
        Number of epochs to train for

    Returns
    -------
    model : keras.models.Model
        Trained keras model
    """
    model.fit([sequence, outseq], sequence, epochs=n_epoch,  verbose=1,
              shuffle=False, callbacks=callbacks_list, validation_split=0.33)
    return model


def get_encoder(model, input_1, encoded):
    encoder = Model(input=input_1, output=encoded)
    return encoder


def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+2)))
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(encoded_input)))
    return decoder


def get_decodings(decoder, encoder, sequence, lms, encodingN, sequence_len):
    encoded = encoder.predict(seq)[-1]
    encoded = RepeatVector(sequence_len)(encoded)
    repeater = np.repeat(encoding, sequence_len, axis=1)
    out_seq = np.reshape(sequence, (len(sequence), sequence_len, 1))
    lms_test = np.reshape(np.repeat(lms[i], sequence_len), (len(sequence), -1))
    out_seq = np.dstack((out_seq, lms_test))
    
    decoding_input = np.concatenate((repeater, out_seq), axis=-1)
    decodings = decoder.predict(decoding_input)
    
    return decodings


def save_model(model, encodingN, LSTMN, model_dir='models/', outdir='./'):
    # make output dir
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".json", "w") as json_file:
        json_file.write(model_json)
    with open(model_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".h5")
    model.save_weights(model_dir+"model.h5")

    logging.info(f'Saved model to {model_dir}')


def save_encodings(model, encoder, sequence, ids, INPUT_FILE,
                   encodingN, LSTMN, N, sequence_len,
                   model_dir='encodings/', outdir='./'):

    # Make output directory
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    encodings = encoder.predict(sequence)
    encoder.reset_states()

    encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'.npz'
    np.savez(encoder_sne_file, encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)
    np.savez(model_dir+'en.npz', encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)

    logging.info(f'Saved encodings to {model_dir}')


def main():
    parser = ArgumentParser()
    parser.add_argument('lcfile', type=str, help='Light curve file')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the LC data (single file)')
    parser.add_argument('--plot', type=bool, default=False, help='Plot LCs')
    parser.add_argument('--neuronN', type=int, default=NEURON_N_DEFAULT, help='Number of neurons in hidden layers')
    parser.add_argument('--encodingN', type=int, default=ENCODING_N_DEFAULT,
                        help='Number of neurons in encoding layer')
    parser.add_argument('--n-epoch', type=int, dest='n_epoch',
                        default=N_EPOCH_DEFAULT,
                        help='Number of epochs to train for')

    args = parser.parse_args()

    sequence, outseq, ids, maxlen, nfilts = prep_input(args.lcfile, save=True, outdir=args.outdir)

    model, callbacks_list, input_1, encoded = make_model(args.neuronN,
                                                         args.encodingN,
                                                         maxlen, nfilts)
    model = fit_model(model, callbacks_list, sequence, outseq, args.n_epoch)
    encoder = get_encoder(model, input_1, encoded)
    
    if args.outdir[-1] != '/':
        args.outdir += '/'
    save_model(model, args.encodingN, args.neuronN, outdir=args.outdir)

    save_encodings(model, encoder, sequence, ids, args.lcfile,
                   args.encodingN, args.neuronN, len(ids), maxlen,
                   outdir=args.outdir)


if __name__ == '__main__':
    main()
