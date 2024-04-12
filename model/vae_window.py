import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Conv1DTranspose, Flatten, Reshape, Lambda
from keras import backend as K


def vae_window(train, latent_dim=8):
    from utils.sample import sample
    from model.VAELossLayer import VAELossLayer
    encoder_inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(encoder_inputs)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    x = layers.Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(train.shape[1]*16, activation='relu')(latent_inputs)
    x = layers.Reshape((int(train.shape[1]/4), 64))(x)
    x = layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu")(x)
    decoder_outputs = layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae_loss_layer = VAELossLayer()([encoder_inputs, outputs, z_mean, z_log_var])
    vae = Model(encoder_inputs, [outputs, vae_loss_layer], name='vae')
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

    return vae