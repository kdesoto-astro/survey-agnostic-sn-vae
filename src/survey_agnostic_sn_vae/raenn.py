# REMADE IN PYTORCH
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Iterator
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.optim import Adam
import logging
import datetime

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

DEFAULT_DEVICE = 'mps' # change to M1 GPU
DEFAULT_HIDDEN = 100
DEFAULT_LATENT = 10
DEFAULT_BATCH = 1024
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4

print(torch.backends.mps.is_available())

    
def contrastive_loss(
    samples,
    means,
    logvars,
    ids,
    distance='wasserstein',
    temp=10.0
):
    """Pushes multiple views of the same
    object together in the latent space.
    """
    N = samples.shape[0]
    # samples is the latent variables
    S_i = torch.unsqueeze(samples, 0).repeat((N,1,1))
    S_j = torch.transpose(S_i, 0, 1)
    
    Z_i = torch.unsqueeze(means, 0).repeat((N,1,1))
    Z_j = torch.transpose(Z_i, 0, 1)
    
    Sig_i = torch.unsqueeze(logvars, 0).repeat((N,1,1))
    Sig_j = torch.transpose(Sig_i, 0, 1)
    stddev1 = torch.exp(0.5*Sig_i)
    stddev2 = torch.exp(0.5*Sig_j)
    
    # make "adjacency matrix" type thing for object IDs
    objid_mat = torch.unsqueeze(ids, 0).repeat((N,1))
    objid_bool_mat = torch.eq(objid_mat, torch.transpose(objid_mat,0,1))

    # Distance for object IDs is 0 if they're the same and 1 otherwise
    objid_dist = objid_bool_mat.type(torch.float32)
    objid_dist[range(N), range(N)] = 0.0 # unset diagonal
    
    # check rows where there's NO matches
    no_match_idxs = torch.all(objid_dist == 0.0, dim=0)
    
    # inverse identity matrix
    denom_arr = 1. - torch.eye(N).to(samples.device)
    
    # SOFT NEAREST NEIGHBORS LOSS
    if distance == 'cosine':
        cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        dists = 1 - cos_sim(S_i, S_j)
    
    elif distance == 'cosine_means':
        cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        dists = 1 - cos_sim(Z_i, Z_j)
        
    elif distance == 'euclidean':
        dists = torch.norm(S_i - S_j, dim=-1)
        dists = torch.clamp(dists, min=1e-8)
        
    elif distance == 'euclidean_means':
        dists = torch.norm(Z_i - Z_j, dim=-1)
        dists = torch.clamp(dists, min=1e-8)
        
    elif distance == 'kl':
        kl = torch.log(stddev2/stddev1) + (stddev1**2 + (Z_i - Z_j).pow(2)) / (2*stddev2**2) - 0.5
        kl[kl > 1000.] = 1000.
        dists = torch.mean(kl, dim=-1)
        
    elif distance == 'mahalonobis':
        dists = torch.norm((S_i - Z_j)/ stddev2, dim=-1)
        dists = torch.clamp(dists, max=50.0)
        
    elif distance == 'wasserstein':
        w_squared = torch.norm(Z_i - Z_j, dim=-1)**2 + torch.sum(stddev1**2 + stddev2**2 - 2*stddev1*stddev2, dim=-1)
        dists = w_squared
        dists[dists > 100.] = 100.
        
    else:
        raise ValueError(f"distance metric {distance} not implemented!")
    
    exp_sims = torch.exp(-dists / temp)
    num = torch.sum(
        (objid_dist * exp_sims)[~no_match_idxs][:,~no_match_idxs],
        dim=1
    )
    denom = torch.sum(
        (denom_arr * exp_sims)[~no_match_idxs][:,~no_match_idxs],
        dim=1
    )
    ratio = torch.log(num / denom)
    
    l = -1 * torch.mean(ratio)

    return l
        
    
def loss_function(
    y_true, y_pred, loss_mask, nfilts,
    mean, log_var, samples, ids,
    add_kl=False, add_contrastive=False,
    metric=None, temp=None,
):
    # mask out 
    f_true = y_true[:,:,1:nfilts+1]
    err_true = y_true[:,:,1+nfilts:1+2*nfilts]
    
    mean_per_lc = torch.mean(
        torch.square((f_true - y_pred)/err_true)[~loss_mask]
    )
    
    recon_loss = torch.mean(mean_per_lc)
    losses = [recon_loss,]

    if add_kl:
        kl_loss = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        losses.append(kl_loss)
        
    if add_contrastive:
        cl = contrastive_loss(
            samples, mean, log_var, ids,
            distance=metric, temp=temp
        )
        losses.append(cl)

    return losses


class CustomBatchSampler(Sampler):
    def __init__(self, data, batch_size: int, shuffle: bool, device: str) -> None:
        self.data = data
        self.batch_size = batch_size
        self.generator = None
        self.unique_ids, self.id_groupings = torch.unique(
            data.ids, sorted=True, return_inverse=True
        )
        self.shuffle = shuffle
        self.device = device

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        n = len(self.data)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
            
        if not self.shuffle:
            all_ids = torch.arange(n)
            
        else:
            shuffled_id_idxs = torch.randperm(
                len(self.unique_ids), generator=generator
            )
            all_ids = torch.empty((0, 1), dtype=torch.long).to(self.device)

            for i in shuffled_id_idxs:
                all_ids = torch.cat((all_ids, torch.argwhere(self.id_groupings == i)))
            all_ids = torch.squeeze(all_ids)
                    
        for batch in all_ids.chunk(len(self)):
            yield batch.tolist()
            

class SNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sequence, outseq, obj_ids, loss_mask, device):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input1 = torch.from_numpy(sequence).to(device)
        self.input2 = torch.from_numpy(outseq).to(device)
        self.ids = torch.from_numpy(obj_ids).to(device)
        self.mask = torch.from_numpy(loss_mask).bool().to(device)

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.input1[idx], self.input2[idx], self.ids[idx], self.mask[idx]
    

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
    
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
    
class VAE(nn.Module):

    def __init__(
        self,
        input_shape,
        hidden_dim=DEFAULT_HIDDEN,
        latent_dim=DEFAULT_LATENT,
        device=DEFAULT_DEVICE
    ):
        super(VAE, self).__init__()

        self.device = device
        self.maxlen = input_shape[1]
        self.input_dim = input_shape[2]
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # encoder
        self.encoder = nn.Sequential(
            TimeDistributed(nn.Linear(self.input_dim, hidden_dim), batch_first=True),
            nn.LeakyReLU(0.01),
            nn.GRU(hidden_dim, hidden_dim, batch_first=True),
            SelectItem(1),
            SelectItem(0),
            nn.LeakyReLU(0.01)
        )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.out_dim = latent_dim + 4
        # decoder
        self.decoder = nn.Sequential(
            TimeDistributed(nn.Linear(self.out_dim, hidden_dim), batch_first=True),
            nn.LeakyReLU(0.01),
            TimeDistributed(nn.Linear(hidden_dim, 1), batch_first=True),
            nn.LeakyReLU(0.01),
        )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def merge(self, z, x2):
        z_reshape = torch.reshape(z, (z.shape[0], 1, -1))
        z_repeat = z_reshape.repeat((1, self.maxlen*6, 1))
        return torch.concatenate([z_repeat, x2], axis=-1)
        
    def reshape_decoding(self, x):
        x_reshape = torch.reshape(x, (-1, self.maxlen, 6))
        return x_reshape
        
    def forward(self, x1, x2):
        mean, logvar = self.encode(x1)
        stddev = torch.exp(0.5*logvar)
        z = self.reparameterization(mean, stddev)
        
        merged = self.merge(z, x2)
        x_hat = self.decode(merged)
        x_reshape = self.reshape_decoding(x_hat)
        
        return x_reshape, z, mean, logvar
    
    def save(self, outdir='./', model_dir='models/'):
        model_dir = os.path.join(outdir, model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(
            self, os.path.join(
                model_dir,
                f"model_{date}_{self.latent_dim}_{self.hidden_dim}.pt"
            )
        )
        torch.save(self, os.path.join(model_dir, "model.pt"))

        logging.info(f'Saved model to {model_dir}')
        
    def save_outputs(self, dataset, ids=None, outdir='./', model_dir='outputs/'):
        # Make output directory
        model_dir = os.path.join(outdir, model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=256,
            shuffle=False,
        )
            
        decodings, z_means, z_logvars = evaluate(self, data_loader)
        
        decodings = decodings.cpu()
        z_means = z_means.cpu()
        z_logvars = z_logvars.cpu()
        
        np.savez(
            os.path.join(
                model_dir,
                f"out_{date}_{self.latent_dim}_{self.hidden_dim}.npz"
            ), z_means=z_means, z_logvars=z_logvars, decodings=decodings, ids=ids
        )
        np.savez(
            os.path.join(
                model_dir, "out.npz"
            ), z_means=z_means, z_logvars=z_logvars, decodings=decodings, ids=ids
        )
        
        logging.info(f'Saved outputs to {model_dir}')
    

def evaluate(model, data_loader):
    
    x_hats, means, log_vars = None, None, None
    
    with torch.no_grad():
        model.eval()
        for (x1, x2, x3, x4) in data_loader:
            x_hat, z, mean, log_var = model(x1, x2)
            
            if x_hats is None:
                x_hats = x_hat
                means = mean
                log_vars = log_var
            else:
                x_hats = torch.cat((x_hats, x_hat))
                means = torch.cat((means, mean))
                log_vars = torch.cat((log_vars, log_var))     
            
    return x_hats, means, log_vars


def train(
    model, optimizer, train_loader,
    test_loader, nfilts, epochs,
    add_kl=True, add_contrastive=True,
    metric=None, temp=None,
    latent_space_plot_dir=None,
):
    if latent_space_plot_dir is not None:
        from survey_agnostic_sn_vae.plotting import plot_latent_space
        
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x1, x2, x3, x4) in enumerate(train_loader):

            optimizer.zero_grad()

            x_hat, z, mean, log_var = model(x1, x2)
            losses = loss_function(
                x1, x_hat, x4, nfilts,
                mean, log_var, z, x3,
                add_kl=add_kl, add_contrastive=add_contrastive,
                metric=metric, temp=temp
            )
            loss = sum(losses)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        # test loop
        with torch.no_grad():
            test_loss = 0
            model.eval()
            
            means = torch.empty((0,model.latent_dim), dtype=torch.float32)
            logvars = torch.empty((0,model.latent_dim), dtype=torch.float32)
            samples = torch.empty((0,model.latent_dim), dtype=torch.float32)
            all_ids = torch.empty((0,), dtype=torch.float32)
            
            for test_batch_idx, (x1, x2, x3, x4) in enumerate(test_loader):
                x_hat, z, mean, log_var = model(x1, x2)
                test_losses = loss_function(
                    x1, x_hat, x4, nfilts,
                    mean, log_var, z, x3,
                    add_kl=add_kl, add_contrastive=add_contrastive,
                    metric=metric, temp=temp
                )
                loss2 = sum(test_losses)
                test_loss += loss2.item()
                means = torch.cat((means, mean), axis=0)
                logvars = torch.cat((logvars, log_var), axis=0)
                samples = torch.cat((samples, z), axis=0)
                all_ids = torch.cat((all_ids, x3), axis=0)
                
        
            if latent_space_plot_dir is not None:
                save_fn = os.path.join(
                    latent_space_plot_dir, f'{epoch}'.zfill(4) + '.pdf'
                )
                plot_latent_space(
                    means, logvars, samples,
                    all_ids, save_fn,
                    show_contrastive=add_contrastive
                )
                
            

        if epoch % 10 == 0:
            print(
                "\tEpoch",
                epoch + 1,
                "\tTrain Loss: ",
                train_loss/len(train_loader),
                "\tVal Loss: ",
                test_loss/len(test_loader)
            )
            print(
                '\tTrain',
                [x.item() for x in losses],
                '\tTest',
                [x.item() for x in test_losses]
            )
                
    return train_loss/len(train_loader), test_loss/len(test_loader)


def fit_model(
    model, sequence, outseq,
    loss_mask,
    n_epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LR,
    batch_size=DEFAULT_BATCH,
    device=DEFAULT_DEVICE,
    add_kl=True,
    add_contrastive=True,
    metric=None, temp=None,
    latent_space_plot_dir=None,
):
    if latent_space_plot_dir is not None:
        os.makedirs(latent_space_plot_dir, exist_ok=True)
        
    seq_ids = sequence[:,0,-1]
    sequence = sequence[:,:,:-1]
    
    nfilts = int((sequence.shape[-1] - 1) / 4) # TODO: change for filter width inclusion
    input_dim = sequence.shape[1]
    
    (
        train_seq, test_seq,
        train_out, test_out,
        train_id, test_id,
        train_mask, test_mask
    ) = train_test_split(
        sequence, outseq,
        seq_ids, loss_mask,
        shuffle=False,
        test_size=0.2
    )
    
    train_dataset = SNDataset(train_seq, train_out, train_id, train_mask, device)
    test_dataset = SNDataset(test_seq, test_out, test_id, test_mask, device)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=CustomBatchSampler(
            data=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            device=device,
        ),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=CustomBatchSampler(
            data=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            device=device
        ),
    )
    # TODO: set device = M1 gpu
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train(
        model, optimizer, train_loader, test_loader, nfilts,
        epochs=n_epochs, add_kl=add_kl, add_contrastive=add_contrastive,
        metric=metric, temp=temp,
        latent_space_plot_dir=latent_space_plot_dir
    )
    
    return model
    

    