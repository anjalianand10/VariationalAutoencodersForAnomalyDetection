from keras.models import Model
from keras.layers import Input
from model.encoder import encoder
from model.decoder import decoder
from model.VAELossLayer import VAELossLayer

def VAEModel(input_dim, intermediate_dim, latent_dim, beta=0.05):

    input_shape = (input_dim,)
    inputs = Input(shape=input_shape, name='encoder_input')

    # Build the encoder
    enc, z_mean, z_log_var = encoder(input_shape, intermediate_dim, latent_dim)

    # Build the decoder
    dec = decoder(latent_dim, intermediate_dim, input_dim)

    # Build the VAE loss layer
    outputs = dec(enc(inputs))
    vae_loss_layer = VAELossLayer()([inputs, outputs, z_mean, z_log_var, beta])

    vae_model = Model(inputs, [outputs, vae_loss_layer], name='vae_mlp') 
    return vae_model