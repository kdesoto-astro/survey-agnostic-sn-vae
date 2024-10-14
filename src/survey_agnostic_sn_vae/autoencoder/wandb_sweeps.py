import datetime
import os

import wandb
import h5py
import numpy as np
import equinox as eqx
import yaml
import jax

jax.config.update('jax_platform_name', 'cpu')

from survey_agnostic_sn_vae.autoencoder.raenn_equinox import VAE, fit_model
from survey_agnostic_sn_vae.autoencoder.vae_plotting import plot_latent_space, plot_decodings

now = datetime.datetime.now()
DATE = str(now.strftime("%Y-%m-%d"))


def import_config_yaml(config_fn):
    """Import and validate config yaml."""
    with open(config_fn, 'r') as file:
        config_dict = yaml.safe_load(file)
    for val in config_dict.values():
        if 'value' in val:
            assert len(val) == 1
        elif 'distribution' in val:
            assert len(val) > 1
        elif isinstance(val, list):
            assert len(val) > 1
        else:
            print(val, type(val))
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
    checkpoint_model_fn=None,
    num_runs=5,
    sweep_name="sweep",
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

        print(sweep_name+suffix)
        with wandb.init(
            name=sweep_name+suffix,
            config=config
        ):
            config = wandb.config
            
            # load data
            with h5py.File(data_fn, 'r') as file:
                encoder_inputs = file['encoder_input'][:]
                ids = file['ids'][:]
                classes = file['classes'][:]
                num_samples = len(encoder_inputs)
                train_encoder_inputs = encoder_inputs[:num_samples // 10 * 9]
                val_encoder_inputs = encoder_inputs[num_samples // 10 * 9:]
                train_ids = ids[:num_samples // 10 * 9]
                val_ids = ids[num_samples // 10 * 9:]
                train_classes = classes[:num_samples // 10 * 9]
                val_classes = classes[num_samples // 10 * 9:]

            vae_config_keys = {
                'hidden_dim', 'out_dim'
            }
            vae_config = {k:val for k, val in config.items() if k in vae_config_keys}
            
            fit_config_keys = {
                'batch_size', 'num_epochs', 'learning_rate',
                'include_reconstructive', 'include_kl', 'contrastive_params',
                'transfer_learning'
            }

            fit_config = {k:val for k, val in config.items() if k in fit_config_keys}
            
            if checkpoint_model_fn:
                model = eqx.filter_eval_shape(VAE, **vae_config)
                model = eqx.tree_deserialise_leaves(checkpoint_model_fn, model)

            else:
                model = VAE(**vae_config)

            model, _, val_loss = fit_model(
                model=model,
                encoder_inputs=train_encoder_inputs,
                ids=train_ids,
                val_encoder_inputs=val_encoder_inputs,
                val_ids=val_ids,
                **fit_config,
                wandb_log=True,
            )
            wandb.summary['best_val_loss'] = np.min(val_loss)
            wandb.summary['best_epoch'] = np.argmin(val_loss)

            save_path = os.path.join(save_dir, "vae"+sweep_name+suffix+".eqx")
            eqx.tree_serialise_leaves(save_path, model)

            # save model as artifact
            artifact = wandb.Artifact(name = "vae"+sweep_name+suffix, type = "model")
            artifact.add_file(save_path)
            artifact.save()

            # save latent space + decoding images to wandb
            fig, _ = plot_latent_space(
                encoder_inputs[:256],
                ids[:256],
                model,
                classes=train_classes[:256]
            )
            latent_path = os.path.join(save_dir, "latent_train"+sweep_name+suffix+".png")
            fig.savefig(latent_path)
            wandb.log({"latent_space": wandb.Image(latent_path)})

            fig, _ = plot_latent_space(
                val_encoder_inputs[:256],
                val_ids[:256],
                model,
                classes=val_classes[:256],
            )
            latent_path = os.path.join(save_dir, "latent_val"+sweep_name+suffix+".png")
            fig.savefig(latent_path)
            wandb.log({"latent_space": wandb.Image(latent_path)})


            for i, (fig, _) in enumerate(plot_decodings(
                encoder_inputs, ids,
                model, count=3
            )):  
                decoding_path = os.path.join(save_dir, "decodings"+sweep_name+suffix+f"_{i}.png")
                fig.savefig(decoding_path)
                wandb.log({"decodings": wandb.Image(decoding_path)})


    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, single_run, count=num_runs)