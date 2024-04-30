# REMADE IN PYTORCH
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
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

def loss_function(y_true, y_pred, nfilts, mean, log_var):
    f_true = torch.narrow(y_true, 2, 1, nfilts)
    err_true = torch.narrow(y_true, 2, nfilts+1, nfilts)
    err_padding = torch.max(err_true[:,-1,-1])
    tmp = torch.max(err_true, 2).values
    idx_padding = torch.greater_equal(
        tmp, err_padding * 0.9
    ) # no more padding
    idx_padding_reshaped = torch.unsqueeze(idx_padding, 2).repeat((1,1,nfilts))
    reduced_mean = torch.mean(
        torch.square((f_true - y_pred)/err_true)[idx_padding_reshaped]
    )
    loss = 0.5 * reduced_mean # way overweight
        
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return loss + KLD


class SNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sequence, outseq, device):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input1 = torch.from_numpy(sequence).to(device)
        self.input2 = torch.from_numpy(outseq).to(device)

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.input1[idx], self.input2[idx]
    

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
            nn.LeakyReLU(0.2),
            nn.GRU(hidden_dim, hidden_dim, batch_first=True),
            SelectItem(1),
            SelectItem(0),
            nn.LeakyReLU(0.2)
        )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.out_dim = latent_dim + 3
        # decoder
        self.decoder = nn.Sequential(
            TimeDistributed(nn.Linear(self.out_dim, hidden_dim), batch_first=True),
            nn.LeakyReLU(0.2),
            TimeDistributed(nn.Linear(hidden_dim, 1), batch_first=True),
            nn.LeakyReLU(0.2),
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

    def forward(self, x1, x2):
        mean, logvar = self.encode(x1)
        stddev = torch.exp(0.5*logvar)
        z = self.reparameterization(mean, stddev)
        
        # concat with x2
        z = torch.reshape(z, (z.shape[0], 1, -1))
        z_repeat = z.repeat((1, self.maxlen*6, 1))
        merged = torch.concatenate([z_repeat, x2], axis=-1)
        x_hat = self.decode(merged)
        x_reshape = torch.reshape(x_hat, (-1, self.maxlen, 6))
        return x_reshape, mean, logvar
    
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
            batch_size=64,
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
        for (x1, x2) in data_loader:
            x_hat, mean, log_var = model(x1, x2)
            
            if x_hats is None:
                x_hats = x_hat
                means = mean
                log_vars = log_var
            else:
                x_hats = torch.cat((x_hat, x_hats))
                means = torch.cat((means, mean))
                log_vars = torch.cat((log_vars, log_var))     
            
    return x_hats, means, log_vars


def train(
    model, optimizer, train_loader,
    test_loader, nfilts, epochs,
):
    train_len = len(train_loader.dataset)
    test_len = len(test_loader.dataset)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x1, x2) in enumerate(train_loader):
            #x1 = x1.to(device, non_blocking=True)
            #x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x1, x2)
            loss = loss_function(x1, x_hat, nfilts, mean, log_var)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        # test loop
        with torch.no_grad():
            test_loss = 0
            model.eval()
            for test_batch_idx, (x1, x2) in enumerate(test_loader):
                #x1 = x1.to(device, non_blocking=True)
                #x2 = x2.to(device, non_blocking=True)
                x_hat, mean, log_var = model(x1, x2)
                loss = loss_function(x1, x_hat, nfilts, mean, log_var)
                test_loss += loss.item()

        print(
            "\tEpoch",
            epoch + 1,
            "\tTrain Loss: ",
            train_loss/train_len,
            "\tVal Loss: ",
            test_loss/test_len,
        )
    return train_loss/train_len, test_loss/test_len


def fit_model(
    model, sequence, outseq,
    n_epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LR,
    batch_size=DEFAULT_BATCH,
    device=DEFAULT_DEVICE
):
    seq_ids = sequence[:,:,-1]
    sequence = sequence[:,:,:-1]
    
    nfilts = int((sequence.shape[-1] - 1) / 3) # TODO: change for filter width inclusion
    input_dim = sequence.shape[1]
    
    train_seq, test_seq, train_out, test_out = train_test_split(sequence, outseq, test_size=0.2)
    train_dataset = SNDataset(train_seq, train_out, device)
    test_dataset = SNDataset(test_seq, test_out, device)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    # TODO: set device = M1 gpu
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train(
        model, optimizer, train_loader, test_loader, nfilts,
        epochs=n_epochs
    )
    
    return model
    

    