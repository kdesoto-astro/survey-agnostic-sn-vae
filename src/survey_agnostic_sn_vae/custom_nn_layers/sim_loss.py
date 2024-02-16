import tensorflow as tf
from tf.keras.layers import Layer


class SimilarityLossLayer(Layer):
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
    def __init__(self, maxlen, bmin, bmax, **kwargs):

        self.k1 = 20. #TODO: add cyclical annealing
        self.k0 = 2.

        super(SimilarityLossLayer, self).__init__(**kwargs)

    def call(self, samples, input1):
        # samples is the latent variables
        # input1 is the entire input array (last row is object IDs)
        objids = input1[:,:,-1]

        # calculate distance in latent space
        # calculate what the "distances" are supposed to be
        # get the difference
        # average over elements
        S_i = tf.repeat(tf.expand_dims(samples, 0), tf.shape(samples)[0], axis=0)
        S_j = tf.transpose(S_i, perm=[1,0,2])
        # This is the distance in latent space
        S_ij = tf.reduce_mean(tf.math.square(S_i - S_j), axis=-1)
        # make "adjacency matrix" type thing for object IDs
        objid_mat = tf.repeat(tf.expand_dims(objids, 0), tf.shape(objids)[0], axis=0)

        objid_bool_mat = tf.math.logical_not(tf.math.equal(objid_mat, tf.transpose(objid_mat)))

        # Distance for object IDs is 0 if they're the same and 1 otherwise
        objid_dist = tf.tensor(1.) * objid_bool_mat

        # Compute distance between distances?
        L_sim = tf.reduce_mean(tf.math.square(S_ij - objid_dist), axis=-1)

        self.add_metric(L_sim, "constrastive_loss")

        return samples
