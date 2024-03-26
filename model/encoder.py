from keras.models import Model
from keras.layers import Lambda, Input, Dense
from utils.sample import sample

def encoder(input_shape, intermediate_dim, latent_dim):
    #
    # Encoder model for VAE
    #
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use the reparameterization trick and get the output from the sample() function
    z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs, z, name='encoder')
    return [encoder, z_mean, z_log_var]