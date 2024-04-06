import tensorflow as tf

class VAELossLayer(tf.keras.layers.Layer):
    #
    # Custom layer for VAE loss
    #
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var, beta = inputs
        x = tf.reshape(x, tf.shape(x_decoded_mean))
        reconstruction_loss = tf.reduce_sum(tf.square(x - x_decoded_mean))
        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        # return the average loss 
        total_loss = tf.reduce_mean(reconstruction_loss + beta * kl_loss)
        self.add_loss(total_loss)
        return x  # Dummy output