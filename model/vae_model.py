import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import optimizers

def vae_model(train):
    from utils.functions import sample
    from model.VAELossLayer import VAELossLayer

    original_dim = train.shape[1]
    input_shape = (original_dim,)
    intermediate_dim = int(original_dim / 2)
    latent_dim = int(original_dim / 3)

    inputs = Input(shape=input_shape, name='encoder_input')

    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    x = Dropout(0.5)(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, z, name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs))
    vae_loss_layer = VAELossLayer()([inputs, outputs, z_mean, z_log_var])

    vae_model = Model(inputs, [outputs, vae_loss_layer], name='vae_mlp')
    opt = optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)

    vae_model.compile(optimizer=opt, loss=lambda y_true, y_pred: y_pred)
    
    return vae_model