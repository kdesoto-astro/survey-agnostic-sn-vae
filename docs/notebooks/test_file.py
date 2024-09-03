from survey_agnostic_sn_vae.preprocessing import prep_input
from survey_agnostic_sn_vae.archival.raenn import *
from survey_agnostic_sn_vae.archival.kl_loss import SamplingLayer
from survey_agnostic_sn_vae.archival.recon_loss import ReconstructionLoss
import pretty_errors
import time
from torch.profiler import profile, record_function, ProfilerActivity

pretty_errors.configure(
    line_color = pretty_errors.BRIGHT_RED,
    exception_color = pretty_errors.BRIGHT_MAGENTA,
    exception_arg_color = pretty_errors.CYAN,
    exception_file_color = pretty_errors.RED_BACKGROUND + pretty_errors.BRIGHT_WHITE,
    display_locals = True,
)


OUTDIR = 'superraenn'
LCFILE = 'superraenn/lcs_ALL.npz'

device = 'mps'
batch_size=1024

sequence, outseq, ids, maxlen, nfilts = prep_input(LCFILE, save=True, outdir=OUTDIR)
model = VAE(sequence[:,:,:-1].shape, device=device)
model = fit_model(model, sequence, outseq, n_epochs=5, device=device, batch_size=batch_size)
model.save(outdir=OUTDIR)
dataset = SNDataset(sequence[:,:,:-1], outseq, device=device)
model.save_outputs(
    dataset, ids=ids, outdir=OUTDIR
)
"""
device = 'mps'
test_x1 = torch.from_numpy(sequence[:1024,:,:-1]).to(device=device)
test_x2 = torch.from_numpy(outseq[:1024]).to(device=device)

with profile() as prof:
    model = VAE(sequence[:,:,:-1].shape, device=device).to(device=device)
    model(test_x1, test_x2)
    
print(prof.key_averages().table())


for device in ['cpu', 'mps']:
    for batch_size in [2048, 4096, 8192, 16384]:
        start_time = time.time()
        model = VAE(sequence[:,:,:-1].shape, device=device)
        model = fit_model(model, sequence, outseq, n_epochs=5, device=device, batch_size=batch_size)
        print(device, batch_size, time.time() - start_time)
"""

"""
encoder = get_encoder(model, input_1, encoded)

if OUTDIR[-1] != '/':
    OUTDIR += '/'
    
save_model(model, ENCODING_N_DEFAULT, NEURON_N_DEFAULT, outdir=OUTDIR)

save_encodings( 
    model, encoder, sequence, ids, LCFILE,
    ENCODING_N_DEFAULT, NEURON_N_DEFAULT, len(ids), maxlen,
    OUTDIR
)
"""