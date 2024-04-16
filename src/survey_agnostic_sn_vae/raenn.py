# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Model, clone_model
from keras.layers import Input, TimeDistributed
from keras.layers import Dense, GRU, concatenate, Concatenate, JaxLayer
from keras.layers import RepeatVector, Reshape
from keras.optimizers import Adam
from keras import Sequential, ops
from keras import callbacks as callbacks_module

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping
import datetime
import os
import logging
from keras.utils import set_random_seed
import keras
import pretty_errors
import math
import jax

from survey_agnostic_sn_vae.preprocessing import prep_input
#from survey_agnostic_sn_vae.custom_nn_layers.sim_loss import SimilarityLossLayer
from survey_agnostic_sn_vae.custom_nn_layers.kl_loss import SamplingLayer, AnnealingCallback
from survey_agnostic_sn_vae.custom_nn_layers.recon_loss import ReconstructionLoss
from survey_agnostic_sn_vae.custom_nn_layers.jax_gru import GRUHaiku
from survey_agnostic_sn_vae.custom_nn_layers.custom_jax_model import JAXModel
from survey_agnostic_sn_vae.custom_nn_layers.jax_epoch_iterator import JAXEpochIterator
from survey_agnostic_sn_vae.custom_nn_layers import tree, array_slicing, data_adapter_utils



from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
#import jax
#jax.config.update('jax_disable_jit', False)

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))
set_random_seed(42)

def make_model(LSTMN, encodingN, maxlen, nfilts, n_epochs, batch_size):
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

    print("making model")
    input_1 = Input((None, nfilts*3+1))
    input_2 = Input((None, 1))
    input_3 = Input((maxlen*6, 3))

    gru_layer = JaxLayer(
        **GRUHaiku(LSTMN)
    )
    # make encoder and decoder models separately
    encoder = Sequential()
    encoder.add(
        TimeDistributed(
            Dense(
                LSTMN,
                activation="leaky_relu",
                kernel_initializer=keras.initializers.RandomNormal(stddev=1e-2),
            ),
            name="enc1"
        )
    )

    encoder.add(gru_layer)

    
    # DECODER
    decoder = Sequential()
    decoder.add(
        #TimeDistributed(
        Dense(
            LSTMN,
            activation="leaky_relu",
            kernel_initializer=keras.initializers.RandomNormal(stddev=1e-2)
        ), #name="dec1"
        #)
    )
    decoder.add(
        #TimeDistributed(
        Dense(
            1,
            #activation="leaky_relu",
            #kernel_initializer=keras.initializers.RandomNormal(stddev=1e-2)
        ), #name="dec2"
        #)
    )
    sampling = SamplingLayer()
    annealing = AnnealingCallback(sampling.beta,"cyclical",n_epochs)    
    encoded_mean_layer = Dense(
        encodingN, activation='linear', name="mu",
        kernel_initializer=keras.initializers.RandomNormal(stddev=1e-1),
    )
    
    encoded_log_var_layer = Dense(
        encodingN, activation='linear', name="sigma",
        kernel_initializer=keras.initializers.RandomNormal(stddev=1e-2),
    )
    #encoded = Dense(6)(input_1)
    encoded = encoder(input_1)

    # add KL loss
    #encoded_mean = encoded_mean_layer(encoded)
    #encoded_log_var = encoded_log_var_layer(encoded)
    #encoded_sample = sampling([encoded_mean, encoded_log_var], add_loss=True)
    
    # This just outputs the same input, but adds a loss term
    #encoded = SimilarityLossLayer()(encoded_sample, input_2)
    repeater = RepeatVector(maxlen*6)(encoded)[:,:183*6]
    """
    merged = concatenate([repeater, input_3], axis=-1)
    """
    decoded = decoder(repeater)
    decoded = Reshape((-1,maxlen,6))(decoded)
    #input_modified = input_1[:, :, 1:nfilts+1]
    
    model = JAXModel([input_1, input_2, input_3], decoded)

    new_optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
    
    rl = ReconstructionLoss(nfilts)
    model.compile(
        optimizer=new_optimizer,
        loss=rl,
        metrics=[
            #sampling.kl_loss,
            #sampling.beta,
        ]
    )

    #es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
    #                   verbose=0, mode='min', baseline=None,
    #                   restore_best_weights=True)

    callbacks_list = []#annealing,]
    return model, callbacks_list, input_1, None# encoded


def fit_model(model, callbacks_list, sequence, outseq, n_epoch, batch_size):
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
    seq_ids = sequence[:,:,-1]
    sequence = sequence[:,:,:-1]

    model.fit(
        [sequence, seq_ids, outseq], sequence, epochs=n_epoch, verbose=1,
        shuffle=True, callbacks=callbacks_list, validation_split=0.1,
        batch_size=batch_size
    )
    return model

def get_encoder(model, input_1, encoded):
    encoder = JAXModel(input_1, encoded)
    return encoder


def get_decoder(model, encodingN):
    encoded_input = Input(shape=(None, (encodingN+3)))
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]
    decoder = JAXModel(encoded_input, decoder_layer3(decoder_layer2(encoded_input)))
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

"""
def add_encoded_nodes(model_orig, n_epochs, n=1):
    #Add extra nodes to the encoded means and encoded stddevs layers,
    #from a trained model.
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
"""
    
def raenn_main():


    # minimal test
    print("TEST")
    test_input = np.random.normal(size=(10,10,10))

    def test_func(x, y):
        return x, y

    x, y = jax.lax.scan(test_func, 1, test_input)
    print(x, y)
    print("DONE")
    
    model = GRUHaiku(5)
    rng = jax.random.PRNGKey(0)
    #model['init_fn'](rng)
    #print("init")
    #model['call_fn'](test_input)
    gru_layer = JaxLayer(
        **model
    )
    out = gru_layer(test_input)
    print(out)
    
    


if __name__ == '__main__':
    main()
