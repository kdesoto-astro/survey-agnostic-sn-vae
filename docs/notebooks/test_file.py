from survey_agnostic_sn_vae.preprocessing import prep_input
from survey_agnostic_sn_vae.raenn import *
from keras.models import load_model
from survey_agnostic_sn_vae.custom_nn_layers.kl_loss import SamplingLayer
from survey_agnostic_sn_vae.custom_nn_layers.recon_loss import ReconstructionLoss
from keras.optimizers import Adam
import keras
import pretty_errors

OUTDIR = 'superraenn'
LCFILE = 'superraenn/lcs_ALL.npz'

NEURON_N_DEFAULT = 100
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 100
BATCH_SIZE = 128

#raenn_main()

sequence, outseq, ids, maxlen, nfilts = prep_input(LCFILE, save=True, outdir=OUTDIR)
model, callbacks_list, input_1, encoded = make_model(
    NEURON_N_DEFAULT, ENCODING_N_DEFAULT, int(maxlen), nfilts, N_EPOCH_DEFAULT, BATCH_SIZE
)
"""
model = load_model(
    OUTDIR + "/models/model.h5",
    custom_objects = {
        'SamplingLayer': SamplingLayer,
        'ReconstructionLoss': ReconstructionLoss
    },
    compile=False
)
"""
new_optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
rl = ReconstructionLoss(6)
model.compile(optimizer=new_optimizer, loss=rl)
"""
# try adding a new layer!
model, callbacks_list, input_1, encoded = add_encoded_nodes(model, N_EPOCH_DEFAULT, n=1)
ENCODING_N_DEFAULT += 1
"""
model.summary()
for l in model.layers:
    print(l)

#annealing = AnnealingCallback(model.layers[4].beta,"cyclical",N_EPOCH_DEFAULT)    
#callbacks_list = [annealing,]
callbacks_list = []
model = fit_model(model, callbacks_list, sequence, outseq, N_EPOCH_DEFAULT, BATCH_SIZE)
encoder = get_encoder(model, input_1, encoded)

if OUTDIR[-1] != '/':
    OUTDIR += '/'
    
save_model(model, ENCODING_N_DEFAULT, NEURON_N_DEFAULT, outdir=OUTDIR)

save_encodings( 
    model, encoder, sequence, ids, LCFILE,
    ENCODING_N_DEFAULT, NEURON_N_DEFAULT, len(ids), maxlen,
    OUTDIR
)