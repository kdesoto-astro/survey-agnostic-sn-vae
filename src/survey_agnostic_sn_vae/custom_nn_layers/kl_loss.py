import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import Callback

class AnnealingCallback(Callback):
    """
    Copied over from https://github.com/larngroup/KL_divergence_loss/blob/main/annealing_helper_objects.py.
    """
    def __init__(self,beta,name,total_epochs,M=5,R=1.0):
        assert R >= 0. and R <= 1.
        self.beta=beta
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
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))
            
        elif self.name=="cyclical":
            T=self.total_epochs
            tau = epoch % tf.math.ceil(T/self.M) / (T/self.M)
            #f epoch <= math.ceil(T/self.M):
            #    new_value = 0. # first cycle is all 1
            if tau <= self.R:
                new_value = tau / self.R
            else:
                new_value = 1.
                
            #new_value = 1.
            tf.keras.backend.set_value(self.beta,new_value)
            print("\n Current beta: "+str(tf.keras.backend.get_value(self.beta)))
            
            
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
        beta_weight = 0.0
        self.beta = tf.Variable(
            beta_weight,trainable=False,
            name="Beta_annealing",validate_shape=False
        )
        super().__init__(**kwargs)

    def call(self, inputs, add_loss):
        z_mean, z_log_var = inputs
        
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        samples = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        if add_loss:
            #Add regularizer loss
            kl_loss = - 0.5 * tf.reduce_mean(
                1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
            )
            self.add_metric(kl_loss, "KL_loss")
            self.add_loss(self.beta * kl_loss)
            self.add_metric(self.beta, "beta")
        
        return samples