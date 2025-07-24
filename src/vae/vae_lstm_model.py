import os, warnings, sys
from re import T

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import tensorflow as tf
import joblib
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Conv1DTranspose,
    Reshape,
    Input,
    GlobalAvgPool1D,
    Cropping1D,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, RepeatVector, GlobalAveragePooling1D

from vae.vae_base import BaseVariationalAutoencoder, Sampling


class VariationalAutoencoderLSTM(BaseVariationalAutoencoder):
    model_name = "VAE_LSTM"

    def __init__(self, hidden_layer_sizes, **kwargs):
        super(VariationalAutoencoderLSTM, self).__init__(**kwargs)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [50, 100, 200]

        self.hidden_layer_sizes = hidden_layer_sizes
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )
        x = encoder_inputs
        
        x = Bidirectional(LSTM(64, return_sequences=False), name="bidirectional_lstm")(x)
        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name="decoder_input")

        # Project and repeat latent vector over time
        x = Dense(self.encoder_last_dense_dim, activation="relu", name="dec_dense")(decoder_inputs)
        x = RepeatVector(self.seq_len, name="dec_repeat")(x)  # shape: (B, seq_len, D)

        # Replace Conv1DTranspose stack with one BiLSTM
        x = Bidirectional(
            LSTM(units=self.hidden_layer_sizes[-1], return_sequences=True, activation="tanh"),
            name="dec_bilstm"
        )(x)

        # Final projection per time step
        x = TimeDistributed(Dense(self.feat_dim), name="dec_output_projection")(x)

        self.decoder_outputs = x
        decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder

    @classmethod
    def load(cls, model_dir) -> "VariationalAutoencoderConv":
        params_file = [path for path in os.listdir(model_dir) if path.endswith("_parameters.pkl")]
        params_file = params_file[0]
        dict_params = joblib.load(params_file)
        vae_model = VariationalAutoencoderLSTM(**dict_params)
        vae_model.load_weights(model_dir)
        vae_model.compile(optimizer=Adam())
        return vae_model
