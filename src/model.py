"""Proposed TCN-BiLSTM autoencoder, copied from ``Untitled0.ipynb``.

The architecture is reproduced without modification from notebook cells 29 and
30:

* three residual TCN encoder blocks with 32, 32 and 64 filters and dilations
  1, 2 and 4;
* a bidirectional LSTM with 64 units per direction and return_sequences=True;
* a latent bidirectional LSTM with 32 units per direction and
  return_sequences=False;
* RepeatVector back to 100 time steps;
* decoder bidirectional LSTMs with 32 and 64 units per direction;
* two residual TCN decoder blocks with 64 and 32 filters and dilations 4 and 2;
* a TimeDistributed Dense layer with 13 outputs.

No layer, unit count, kernel size, dilation, dropout rate, activation, or
connection is changed. The notebook is a historical artifact and is not
modified; this module only imports the same definitions into the reproducible
pipeline.
"""

from __future__ import annotations

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)


def tcn_residual_block(
    x,
    filters,
    kernel_size=3,
    dilation_rate=1,
    dropout_rate=0.1,
):
    """Residual dilated-convolution block copied from notebook cell 29."""

    shortcut = x

    # First dilated convolution
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
    )(x)

    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    # Second dilated convolution
    y = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
    )(y)

    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    # Match shortcut dimensions if required
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
        )(shortcut)

    # Residual connection
    y = Add()([shortcut, y])
    y = Activation("relu")(y)

    return y


def build_tcn_bilstm_autoencoder(timesteps=100, n_features=13):
    """Build the proposed TCN-BiLSTM autoencoder from notebook cell 30."""

    inputs = Input(
        shape=(timesteps, n_features),
        name="input_sequence",
    )

    # --------------------------------------------------------
    # TCN Encoder
    # --------------------------------------------------------

    x = tcn_residual_block(
        inputs,
        filters=32,
        kernel_size=3,
        dilation_rate=1,
    )

    x = tcn_residual_block(
        x,
        filters=32,
        kernel_size=3,
        dilation_rate=2,
    )

    x = tcn_residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=4,
    )

    # --------------------------------------------------------
    # Bi-LSTM Encoder
    # --------------------------------------------------------

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True,
        )
    )(x)

    # Latent representation
    latent = Bidirectional(
        LSTM(
            32,
            return_sequences=False,
        ),
        name="latent_encoder",
    )(x)

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    x = RepeatVector(timesteps)(latent)

    x = Bidirectional(
        LSTM(
            32,
            return_sequences=True,
        )
    )(x)

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True,
        )
    )(x)

    # --------------------------------------------------------
    # TCN Decoder
    # --------------------------------------------------------

    x = tcn_residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=4,
    )

    x = tcn_residual_block(
        x,
        filters=32,
        kernel_size=3,
        dilation_rate=2,
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(
        inputs,
        outputs,
        name="TCN_BiLSTM_Autoencoder",
    )

    return model
