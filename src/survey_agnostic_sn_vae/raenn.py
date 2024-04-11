# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Model, clone_model
from keras.layers import Input, GRU, TimeDistributed
from keras.layers import Dense, concatenate, Concatenate
from keras.layers import RepeatVector
from tensorflow.keras.optimizers.legacy import Adam
from keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping
import datetime
import os
import logging
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed

from survey_agnostic_sn_vae.preprocessing import prep_input
from survey_agnostic_sn_vae.custom_nn_layers.sim_loss import SimilarityLossLayer
from survey_agnostic_sn_vae.custom_nn_layers.kl_loss import SamplingLayer, AnnealingCallback
from survey_agnostic_sn_vae.custom_nn_layers.recon_loss import ReconstructionLoss

print(tf.config.list_physical_devices('GPU'))

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))
set_random_seed(42)


def make_model(LSTMN, encodingN, maxlen, nfilts, n_epochs):
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
    input_1 = Input((None, nfilts*3+2))
    input_2 = Input((maxlen*6, 3))

    # make encoder and decoder models separately
    encoder = Sequential()
    encoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="leaky_relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2),
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
                activation="leaky_relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2)
            ), name="dec1"
        )
    )
    decoder.add(
        TimeDistributed(
            Dense(
                1,
                activation="leaky_relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2)
            ), name="dec2"
        )
    )
    sampling = SamplingLayer()
    annealing = AnnealingCallback(sampling.beta,"cyclical",n_epochs)    
    encoded_mean_layer = Dense(
        encodingN, activation='linear', name="mu",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-1),
    )
    
    encoded_log_var_layer = Dense(
        encodingN, activation='linear', name="sigma",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-2),
    )
    
    encoded = encoder(input_1[:,:,:-1])
    
    # add KL loss
    encoded_mean = encoded_mean_layer(encoded)
    encoded_log_var = encoded_log_var_layer(encoded)
    encoded_sample = sampling([encoded_mean, encoded_log_var], True)
    
    # This just outputs the same input, but adds a loss term
    #encoded = SimilarityLossLayer()(encoded_sample, input_1)

    repeater = RepeatVector(maxlen*6)(encoded_sample)
    merged = concatenate([repeater, input_2], axis=-1)
    decoded = decoder(merged)
    
    model = Model([input_1, input_2], decoded)

    new_optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    
    rl = ReconstructionLoss(nfilts)
    model.compile(optimizer=new_optimizer, loss=rl)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    callbacks_list = [es, annealing]
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
    model.fit(
        [sequence, outseq], sequence, epochs=n_epoch,  verbose=1,
        shuffle=True, callbacks=callbacks_list, validation_split=0.1,
        batch_size=512
    )
    return model

def get_encoder(model, input_1, encoded):
    encoder = Model(input_1, encoded)
    return encoder


def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+3)))
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer3(decoder_layer2(encoded_input)))
    return decoder


def get_decodings(model, sequence, outseq):
    decodings = model([sequence, outseq])
    return decodings


def save_model(model, encodingN, LSTMN, model_dir='models/', outdir='./'):
    # make output dir
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.save(model_dir+"model_"+date+"_"+str(encodingN)+'_'+str(LSTMN)+".h5")
    model.save(model_dir+"model.h5")

    logging.info(f'Saved model to {model_dir}')


def save_encodings(model, encoder, sequence, ids, INPUT_FILE,
                   encodingN, LSTMN, N, sequence_len,
                   model_dir='encodings/', outdir='./'):

    # Make output directory
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    encodings = encoder(sequence)
    encoder.reset_states()

    encoder_sne_file = model_dir+'en_'+date+'_'+str(encodingN)+'_'+str(LSTMN)+'.npz'
    np.savez(encoder_sne_file, encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)
    np.savez(model_dir+'en.npz', encodings=encodings, ids=ids, INPUT_FILE=INPUT_FILE)

    logging.info(f'Saved encodings to {model_dir}')


def add_encoded_nodes(model_orig, n_epochs, n=1):
    """Add extra nodes to the encoded means and encoded stddevs layers,
    from a trained model."""
    encoder = model_orig.layers[1]
    LSTMN = encoder.layers[0].output.shape[-1]
    encodingN = model_orig.layers[2].output.shape[-1]
    maxlen = int(model_orig.layers[5].output.shape[1] / 6)
    #maxlen = 0
    
    l_weights = [model_orig.layers[0].get_weights(),]
    
    for subl in model_orig.layers[1].layers:
        l_weights.append(
            subl.get_weights()
        )
    
    # expand mu and sigma weights
    mu_weights = model_orig.layers[2].get_weights()
    sig_weights = model_orig.layers[3].get_weights()
    
    mu_weights[0] = tf.concat([mu_weights[0], tf.random.normal((LSTMN, n), stddev=1e-5)], axis=-1)
    mu_weights[1] = tf.concat([mu_weights[1], tf.random.normal([n,], stddev=1e-5)], axis=0)
    sig_weights[0] = tf.concat([sig_weights[0], tf.random.normal((LSTMN, n), stddev=1e-5)], axis=-1)
    sig_weights[1] = tf.concat([sig_weights[1], tf.random.normal([n,], stddev=1e-5)], axis=0)
    
    l_weights.append(mu_weights)
    l_weights.append(sig_weights)
    
    # expand Sequential layer
    dec1_weights = model_orig.layers[8].layers[0].get_weights()
    dec1_weights[0] = tf.concat([dec1_weights[0], tf.random.normal((n, LSTMN), stddev=1e-5)], axis=-2)
    #dec1_weights[1] = tf.concat([dec1_weights[1], tf.random.normal(n)])
    
    l_weights.append(dec1_weights)
    l_weights.append(model_orig.layers[8].layers[1].get_weights())
    
    # make model accomodating for bigger encoded layer
    model, callbacks_list, input_1, encoded = make_model(LSTMN, encodingN+n, maxlen, 6, n_epochs)
    
    # transfer weights!
    #model.layers[0].set_weights(l_weights[0])
    model.layers[1].layers[0].set_weights(l_weights[1])
    model.layers[1].layers[1].set_weights(l_weights[2])
    model.layers[2].set_weights(l_weights[3])
    model.layers[3].set_weights(l_weights[4])
    model.layers[8].layers[0].set_weights(l_weights[5])
    model.layers[8].layers[1].set_weights(l_weights[6])
    
    return model, callbacks_list, input_1, encoded

    
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
