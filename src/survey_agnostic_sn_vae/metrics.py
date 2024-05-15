# Metrics to quantify intermodal and cross-modal encoding
import torch
import os
from survey_agnostic_sn_vae.preprocessing import prep_input
from survey_agnostic_sn_vae.raenn import SNDataset

def weighted_mse_vectorized(y_true, y_pred, mask):
    # mask out 
    f_true = y_true[:,:,1:nfilts+1]
    err_true = y_true[:,:,1+nfilts:1+2*nfilts]
    
    mean_per_lc = torch.mean(
        torch.square((f_true - y_pred)/err_true)[~mask.bool()]
    )
    
    reduced_mean = torch.mean(mean_per_lc)
    return reduced_mean
    
def calc_inter_modal_weighted_mse(model, dataset):
    out, _, _, _ = model.forward(dataset.input1, dataset.input2)
    return weighted_mse_vectorized(dataset.input1, out, dataset.mask).item()


def calc_cross_modal_weighted_mse(model, dataset):
    """If you decode encoded LC using info from
    positive match, how well do you get the match's LC?
    """
    uids = torch.unique(dataset.ids)
    
    mse = 0
    skip_ct = 0
    for uid in uids:
        sub_ds = dataset[dataset.ids == uid]
        N = len(sub_ds[0])
        
        if N < 2:
            skip_ct += 1
            continue
            
        indices = torch.arange(N)
        # Create all possible pairs of indices without repetition
        pairs = torch.transpose(
            torch.vstack([indices.repeat_interleave(N), indices.repeat(N)]), 0, 1
        )
        # Filter out pairs where the indices are the same
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        sub_i1 = sub_ds[0][pairs[:,0]] # inputs
        sub_i2 = sub_ds[1][pairs[:,1]] # for decodings
        sub_init = sub_ds[0][pairs[:,1]] # to compare with decodings
        sub_mask = sub_ds[3][pairs[:,1]]
    
        sub_out, _, _, _ = model.forward(sub_i1, sub_i2)
        mse += weighted_mse_vectorized(sub_init, sub_out, sub_mask)
        
    return mse.item()/(len(uids) - skip_ct)


if __name__ == "__main__":
    LCFILE = '/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/yse/lcs.npz'
    OUTFILE = '/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/yse/outputs/out.npz'
    OUTDIR = '/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/yse'
    PREPFILE = '/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/prep.npz'

    sequence, outseq, loss_mask, ids, maxlen, nfilts = prep_input(
        LCFILE, load=True, outdir=OUTDIR, prep_file=PREPFILE
    )
    device = 'cpu'
    model_fn = os.path.join(
        OUTDIR,
        "/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/yse/models/model.pt"
    )
    model = torch.load(model_fn).to(device)
    model.device = device

    dataset = SNDataset(sequence[:,:,:-1], outseq, sequence[:,0,-1], loss_mask, device=device)
    print(calc_cross_modal_weighted_mse(model, dataset))
    print(calc_inter_modal_weighted_mse(model, dataset))
        
        
            
            
            
    
        
        