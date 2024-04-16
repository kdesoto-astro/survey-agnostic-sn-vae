from keras.layers import Layer
from keras.metrics import Mean
from keras.callbacks import Callback

import jax.numpy as jnp


class AnnealingCallback(Callback):
    """
    Copied over from https://github.com/larngroup/KL_divergence_loss/blob/main/annealing_helper_objects.py.
    """
    def __init__(self,beta,name,total_epochs,M=5,R=1.0):
        assert R >= 0. and R <= 1
        self.beta = beta
        self.name=name
        self.total_epochs=total_epochs
        self.M = M
        self.R = R
    
    def on_epoch_begin(self,epoch,logs={}):
        if self.name=="normal":
            pass
        elif self.name=="monotonic":
            
            new_value=epoch/float(self.total_epochs)
            if new_value > 1:
                new_value = 1

            self.beta = new_value
            
            print("\n Current beta: "+str(self.beta))
            
        elif self.name=="cyclical":
            T=self.total_epochs
            tau = epoch % jnp.ceil(T/self.M) / (T/self.M)
            #f epoch <= math.ceil(T/self.M):
            #    new_value = 0. # first cycle is all 1
            if tau <= self.R:
                new_value = tau / self.R
            else:
                new_value = 1.

            self.beta = new_value
            print("\n Current beta: "+str(self.beta))
            
            
class SamplingLayer(Layer):
    """
    Samples from the latent normal distribution using
    reparametrization to maintain gradient propagation.
    
    Parameters
    ----------
    samp_args : array
        the mean and log sigma values of the latent space
        distribution
        
    Returns
    ----------
    sample : array
        a sampled value from the latent space
    """
    def __init__(self, **kwargs):
        self.beta = 0.0
        self.kl_loss = Mean(name="kl_loss")
        super().__init__(**kwargs)
        
    def call(self, inputs, add_loss=True):
        z_mean, z_log_var = inputs
        #epsilon = keras.backend.random_normal(shape=z_mean.shape)
        samples = z_mean + jnp.exp(0.5 * z_log_var)

        if add_loss:
            #Add regularizer loss
            kl_loss = -0.5 * (
                1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)
            )
            self.add_loss(self.beta * jnp.mean(kl_loss))
            self.kl_loss.reset_state()
            self.kl_loss.update_state(kl_loss)
            
        return samples