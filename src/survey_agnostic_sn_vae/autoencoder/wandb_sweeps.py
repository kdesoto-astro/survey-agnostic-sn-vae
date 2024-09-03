import datetime
import os

import wandb
import h5py
import numpy as np
import equinox as eqx
import yaml

from survey_agnostic_sn_vae.autoencoder.raenn_equinox import VAE, fit_model

now = datetime.datetime.now()
DATE = str(now.strftime("%Y-%m-%d"))


def import_config_yaml(config_fn):
    """Import and validate config yaml."""
    config_dict = yaml.load(config_fn, Loader=yaml.SafeLoader)
    for val in config_dict.values():
        if 'value' in val:
            assert len(val) == 1
        elif 'distribution' in val:
            assert len(val) > 1
        elif isinstance(val, list):
            assert len(val) > 1
        else:
            raise ValueError("Each value in config_dict must provide either a list, value, or distribution.")
    return config_dict

def set_config_params(config_dict, static_dict):
    """Set parameters from config_yaml as static values."""
    config_dict_copy = config_dict.copy()
    for k, static_val in static_dict.items():
        if k not in config_dict_copy:
            continue
        config_dict_copy[k] = {'value': static_val}
    return config_dict_copy

def wandb_sweep(
    config_dir,
    data_fn,
    save_dir,
    num_runs=5,
    project_name="survey_agnostic_vae_test"
):
    """From a configuration directory, run a sweep
    of hyperparameters using W&B."""
    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_log_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = config_dir

    def single_run(config=None):
        """Single W&B run, given a config file.
        Part of a sweep."""
        suffix = f"_{DATE}"
        if config is not None:
            for k, val in config.items():
                suffix += f'_{k}-{val}'

        with wandb.init(
            name='sweep'+suffix,
            config=config
        ):
            config = wandb.config

            # load data
            with h5py.File(data_fn, 'r') as file:
                encoder_inputs = file['encoder_input'][:]
                num_samples = len(encoder_inputs)
                train_encoder_inputs = encoder_inputs[:num_samples // 10 * 9]
                val_encoder_inputs = encoder_inputs[num_samples // 10 * 9:]

            vae_config_keys = {
                'input_dim', 'hidden_dim', 'out_dim'
            }
            vae_config = {k:val for k, val in config.items() if k in vae_config_keys}
            
            fit_config_keys = {
                'batch_size', 'num_epochs', 'learning_rate',
                'include_reconstructive', 'include_kl', 'contrastive_params'
            }

            fit_config = {k:val for k, val in config.items() if k in fit_config_keys}
            
            model = VAE(**vae_config)

            model, _, val_loss = fit_model(
                model=model,
                encoder_inputs=train_encoder_inputs,
                val_encoder_inputs=val_encoder_inputs,
                **fit_config,
                wandb_log=True,
            )
            wandb.summary['best_val_loss'] = np.min(val_loss)
            wandb.summary['best_epoch'] = np.argmin(val_loss)

            save_path = os.path.join(save_dir, "vae"+suffix)
            eqx.tree_serialise_leaves(save_path, model)

            # save model as artifact
            artifact = wandb.Artifact(name = "vae"+suffix, type = "model")
            artifact.add_file(save_path)
            artifact.save()

            # save latent space + decoding images to wandb


    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, single_run, count=num_runs)