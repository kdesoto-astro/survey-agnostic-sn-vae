import jax.numpy as jnp
import keras


class ReconstructionLoss(keras.losses.Loss):
    """
    Custom loss which doesn't use the errors

    Parameters
    ----------
    yTrue : array
        True flux values
    yPred : array
        Predicted flux values
    """
    def __init__(self, nfilts):
        super().__init__()
        self.nfilts = nfilts
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        f_true = y_true[:, :, 1:self.nfilts+1]

        err_padding = jnp.max(y_true[:,-1,self.nfilts+1])
        err_true = y_true[:,:,self.nfilts+1:2*self.nfilts+1]
        idx_padding = jnp.greater_equal(jnp.max(err_true, axis=2), err_padding * 0.9) # no more padding
        idx_padding_reshaped = jnp.repeat(jnp.expand_dims(idx_padding, 2), self.nfilts, axis=2)
        reduced_mean = jnp.mean(
            jnp.square((f_true - y_pred)/err_true) #* (1 - idx_padding_reshaped)
        )
        loss = 0.5 * reduced_mean # way overweight
        return loss
