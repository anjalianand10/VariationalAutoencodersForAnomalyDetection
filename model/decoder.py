from keras.models import Model
from keras.layers import Input, Dense

def decoder(latent_dim, intermediate_dim, original_dim):
    #
    # Decoder model for VAE
    #
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder
