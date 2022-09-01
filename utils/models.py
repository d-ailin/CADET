from keras import backend as K
from keras import layers
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import Reshape, Layer
import keras
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras import regularizers
import math
import tensorflow as tf
import numpy as np


def fmnist_model(img_train):
    num_filters = 64
    enc_in = Input(shape=img_train.shape[1:])
    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
                activation='relu', padding='valid')(enc_in)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
                activation='relu', padding='valid')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                activation='relu', padding='valid')(enc_2)
    enc_4 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                activation='relu', padding='valid')(enc_3)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                    activation='relu', padding='valid')(enc_4)

    # Decoder
    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(encoded)
    dec_conv_2 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_1)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_2)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                                activation='relu', padding='valid')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                                activation='relu', padding='valid')(dec_conv_4)

    dec_out = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                            activation='sigmoid', padding='valid')(dec_conv_5)

    return Model(inputs=enc_in, outputs=dec_out)


def mnist_model(img_train):
    num_filters = 128
    enc_in = Input(shape=img_train.shape[1:])
    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
            activation='tanh')(enc_in)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_2)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                activation='tanh')(enc_3)

    # Decoder
    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                            activation='tanh')(encoded)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_1)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                            activation='tanh')(dec_conv_4)

    dec_out = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                        activation='sigmoid', padding='valid')(dec_conv_5)
    
    return Model(enc_in, dec_out)


def dae_fmnist_model(img_train):
    num_filters = 64
    enc_in = Input(shape=img_train.shape[1:])
    enc_in_noise = layers.GaussianNoise(0.1)(enc_in)
    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
                activation='relu', padding='valid')(enc_in_noise)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
                activation='relu', padding='valid')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                activation='relu', padding='valid')(enc_2)
    enc_4 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                activation='relu', padding='valid')(enc_3)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                    activation='relu', padding='valid')(enc_4)

    # Decoder
    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(encoded)
    dec_conv_2 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_1)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_2)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                                activation='relu', padding='valid')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                                activation='relu', padding='valid')(dec_conv_4)

    dec_out = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                            activation='sigmoid', padding='valid')(dec_conv_5)

    return Model(inputs=enc_in, outputs=dec_out)


def dae_mnist_model(img_train):
    num_filters = 128
    enc_in = Input(shape=img_train.shape[1:])
    enc_in_noise = layers.GaussianNoise(0.1)(enc_in)

    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
            activation='tanh')(enc_in_noise)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_2)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                activation='tanh')(enc_3)

    # Decoder
    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                            activation='tanh')(encoded)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_1)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                            activation='tanh')(dec_conv_4)

    dec_out = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                        activation='sigmoid', padding='valid')(dec_conv_5)
    
    return Model(enc_in, dec_out)


def image_vae_mnist(X, num_filters=128, latent_dim=2):
    num_filters = 128


    enc_in = Input(shape=X.shape[1:])
    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
            activation='tanh')(enc_in)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
            activation='tanh')(enc_2)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                activation='tanh')(enc_3)

    flat_enc = Flatten()(encoded)

    z_mean = Dense(latent_dim, name='z_mean')(flat_enc)
    z_log_var = Dense(latent_dim, name='z_log_var')(flat_enc)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(enc_in, [z_mean, z_log_var, z])
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    dense_inputs = Dense(2*2*128, activation='relu')(latent_inputs)
    reshaped_inputs = Reshape((2,2,128))(dense_inputs)


    # Decoder
    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                            activation='tanh')(reshaped_inputs)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_1)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                            activation='tanh')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                            activation='tanh')(dec_conv_4)

    decoder_outputs = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                        activation='sigmoid', padding='valid')(dec_conv_5)

    decoder = Model(latent_inputs, decoder_outputs)
    decoder.summary()

    model = VAE(encoder, decoder, latent_dim=2, mode='image')
    return model

def image_vae(X, num_filters = 64, latent_dim=2):

    # Encoder
    enc_in = Input(shape=X.shape[1:])
    # enc_in = Input(shape=input_shape)
    enc_1 = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(2, 2), 
                    activation='relu', padding='valid')(enc_in)
    enc_2 = Conv2D(filters=num_filters, kernel_size=(5, 5), 
                    activation='relu', padding='valid')(enc_1)
    enc_3 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                    activation='relu', padding='valid')(enc_2)
    enc_4 = Conv2D(filters=num_filters, kernel_size=(3, 3), 
                    activation='relu', padding='valid')(enc_3)

    encoded = Conv2D(filters=num_filters*2, kernel_size=(3, 3), 
                    activation='relu', padding='valid')(enc_4)
    flat_enc = Flatten()(encoded)

    z_mean = Dense(latent_dim, name='z_mean')(flat_enc)
    z_log_var = Dense(latent_dim, name='z_log_var')(flat_enc)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(enc_in, [z_mean, z_log_var, z])
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    dense_inputs = Dense(2*2*128, activation='relu')(latent_inputs)
    reshaped_inputs = Reshape((2,2,128))(dense_inputs)

    dec_conv_1 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(reshaped_inputs)
    dec_conv_2 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_1)
    dec_conv_3 = Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), 
                                activation='relu', padding='valid')(dec_conv_2)
    dec_conv_4 = Conv2DTranspose(filters=num_filters, kernel_size=(5, 5), 
                                activation='relu', padding='valid')(dec_conv_3)
    dec_conv_5 = Conv2DTranspose(filters=int(math.log(num_filters, 2)), kernel_size=(5, 5), strides=(2, 2), 
                                activation='relu', padding='valid')(dec_conv_4)

    decoder_outputs = Conv2DTranspose(filters=1, kernel_size=(2, 2), 
                                activation='sigmoid', padding='valid')(dec_conv_5)
    decoder = Model(latent_inputs, decoder_outputs)
    decoder.summary()

    model = VAE(encoder, decoder, latent_dim=2, mode='image')
    return model



class VAE(Model):
    def __init__(self, encoder, decoder, latent_dim=2, mode='image', **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        self.n_samples_, self.n_features_ = data.shape[0], np.prod(data.shape[1:])
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )

            # reconstruction_loss = tf.reduce_mean(
            #     keras.losses.mean_squared_error(data, reconstruction)
            # )
            reconstruction_loss *= self.n_features_
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
      if isinstance(data, tuple):
        data = data[0]

      z_mean, z_log_var, z = self.encoder(data)
      reconstruction = self.decoder(z)
      reconstruction_loss = tf.reduce_mean(
          keras.losses.binary_crossentropy(data, reconstruction)
      )

    #   reconstruction_loss = tf.reduce_mean(
    #       keras.losses.mean_squared_error(data, reconstruction)
    #   )
      reconstruction_loss *= self.n_features_
      kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      kl_loss = tf.reduce_mean(kl_loss)
      kl_loss *= -0.5
      total_loss = reconstruction_loss + kl_loss
      return {
          "loss": total_loss,
          "reconstruction_loss": reconstruction_loss,
          "kl_loss": kl_loss,
      }

    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        return reconstruction



class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon