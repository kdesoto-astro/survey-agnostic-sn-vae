# Run perfetto timing for autoencoder routine
import jax
import os
import h5py

from survey_agnostic_sn_vae import VAE, fit_model, import_config_yaml


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
#jax.config.update('jax_disable_jit', True)
jax.config.update("jax_explain_cache_misses", True)

os.environ['XLA_FLAGS'] = '--xla_cpu_enable_xprof_traceme'

def perfetto_autoencoder(
        data_fn, config_fn,
    ):
    """Run perfetto timing for autoencoder routine.
    When running on cluster, first enable perfetto link by running:
    ssh -L 9001:127.0.0.1:9001 <user>@<host>
    """
    config = import_config_yaml(config_fn)
    with h5py.File(data_fn, 'r') as file:
        encoder_inputs = file['encoder_input'][:]
        ids = file['ids'][:]
        num_samples = len(encoder_inputs)
        train_encoder_inputs = encoder_inputs[:num_samples // 10 * 9]
        val_encoder_inputs = encoder_inputs[num_samples // 10 * 9:]
        train_ids = ids[:num_samples // 10 * 9]
        val_ids = ids[num_samples // 10 * 9:]

    vae_config_keys = {
        'hidden_dim', 'out_dim'
    }
    vae_config = {k:val['value'] for k, val in config.items() if k in vae_config_keys}
    
    fit_config_keys = {
        'batch_size', 'num_epochs', 'learning_rate',
        'include_reconstructive', 'include_kl', 'contrastive_params',
        'transfer_learning'
    }

    fit_config = {k:val['value'] for k, val in config.items() if k in fit_config_keys}
    model = VAE(**vae_config)
    print("starting trace")

    #jax.profiler.start_trace("tensorboard")
    fit_model(
        model=model,
        encoder_inputs=train_encoder_inputs[:128],
        ids=train_ids[:128],
        val_encoder_inputs=val_encoder_inputs[:128],
        val_ids=val_ids[:128],
        **fit_config,
        wandb_log=False,
    )
    #jax.profiler.stop_trace()

if __name__ == '__main__':
    config_fn = 'default_config.yaml'
    data_fn = 'preprocessed_augmented.h5'
    perfetto_autoencoder(data_fn, config_fn)