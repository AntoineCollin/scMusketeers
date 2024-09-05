import keras
import numpy as np
import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.models import Model
from keras.regularizers import l1_l2

from .layers import ColwiseMultLayer, GradReverse

advanced_activations = ("PReLU", "LeakyReLU")


class Encoder(keras.Model):
    def __init__(
        self,
        hidden_size,
        hidden_dropout=None,
        activation="relu",
        bottleneck_activation="linear",
        init="glorot_uniform",
        batchnorm=True,
        l1_enc_coef=0,
        l2_enc_coef=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0] * len(hidden_size)
        if type(hidden_dropout) == float:
            hidden_dropout = [hidden_dropout] * len(hidden_size)
        if len(hidden_dropout) == 1:
            hidden_dropout = hidden_dropout * len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []
        center_idx = len(hidden_size)

        for i, (hid_size, hid_drop) in enumerate(
            zip(hidden_size, hidden_dropout)
        ):
            if i != center_idx:
                layer_name = f"enc_{str(i)}"
                stage = "encoder"
            else:
                layer_name = f"center"
                stage = "center"

            self.hidden.append(
                Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=init,
                    kernel_regularizer=l1_l2(l1_enc_coef, l2_enc_coef),
                    name=layer_name,
                )
            )

            if batchnorm:
                self.hidden_batchnorm.append(
                    BatchNormalization(center=True, scale=True)
                )
            else:
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if layer_name == "center":
                act = bottleneck_activation
            else:
                act = activation
            if act in advanced_activations:
                self.hidden_activations.append(
                    keras.layers.__dict__[act](name=f"{layer_name}_act")
                )
            else:
                self.hidden_activations.append(
                    Activation(act, name=f"{layer_name}_act")
                )

            if hid_drop > 0.0:
                self.hidden_dropout.append(
                    Dropout(hid_drop, name=f"{layer_name}_drop")
                )
            else:
                self.hidden_dropout.append(None)

    def call(self, inputs, training=None):
        Z = inputs
        for layer, activation, batchnorm, dropout in zip(
            self.hidden,
            self.hidden_activations,
            self.hidden_batchnorm,
            self.hidden_dropout,
        ):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z, training=training)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z, training=training)
        return Z

    def build_graph(self, dim):
        x = Input(shape=(dim))
        return Model(inputs=[x], outputs=self.call(x))


class Decoder(keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        hidden_dropout,
        activation="relu",
        init="glorot_uniform",
        batchnorm=True,
        **kwargs,
    ):
        """
        hidden_size : A list of the size for the decoder only. Ex : if the size of the total AE are [200,100,10,100,200], the hidden_size
        for the Decoder should be [100,200]
        """
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0] * len(hidden_size)
        if type(hidden_dropout) == float:
            hidden_dropout = [hidden_dropout] * len(hidden_size)
        if len(hidden_dropout) == 1:
            hidden_dropout = hidden_dropout * len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []

        for i, (hid_size, hid_drop) in enumerate(
            zip(hidden_size, hidden_dropout)
        ):
            layer_name = f"dec_{str(i)}"
            stage = "decoder"
            self.hidden.append(
                Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=init,
                    name=layer_name,
                )
            )

            if batchnorm:
                self.hidden_batchnorm.append(
                    BatchNormalization(center=True, scale=True)
                )
            else:
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if activation in advanced_activations:
                self.hidden_activations.append(
                    keras.layers.__dict__[activation](name=f"{layer_name}_act")
                )
            else:
                self.hidden_activations.append(
                    Activation(activation, name=f"{layer_name}_act")
                )

            if hid_drop > 0.0:
                self.hidden_dropout.append(
                    Dropout(hid_drop, name=f"{layer_name}_drop")
                )
            else:
                self.hidden_dropout.append(None)

    def call(self, inputs, training=None):
        Z = inputs
        for layer, activation, batchnorm, dropout in zip(
            self.hidden,
            self.hidden_activations,
            self.hidden_batchnorm,
            self.hidden_dropout,
        ):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z, training=training)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z, training=training)
        return Z

    # def build_graph(self, dim):
    #     x = Input(shape=(dim))
    #     return Model(inputs=[x], outputs=self.call(x))


class Classifier(keras.Model):
    def __init__(
        self,
        num_classes,
        hidden_size,
        hidden_dropout=None,
        prefix="classifier",
        batchnorm=True,
        activation="relu",
        output_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0] * len(hidden_size)
        if type(hidden_dropout) == float:
            hidden_dropout = [hidden_dropout] * len(hidden_size)
        if len(hidden_dropout) == 1:
            hidden_dropout = hidden_dropout * len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []
        for i, (hid_size, hid_drop) in enumerate(
            zip(hidden_size, hidden_dropout)
        ):
            layer_name = f"{prefix}_{str(i)}"
            self.hidden.append(
                Dense(hid_size, activation=None, name=layer_name)
            )

            if batchnorm:
                self.hidden_batchnorm.append(
                    BatchNormalization(center=True, scale=True)
                )
            else:
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if activation in advanced_activations:
                self.hidden_activations.append(
                    keras.layers.__dict__[activation](name=f"{layer_name}_act")
                )
            else:
                self.hidden_activations.append(
                    Activation(activation, name=f"{layer_name}_act")
                )

            if hid_drop > 0.0:
                self.hidden_dropout.append(
                    Dropout(hid_drop, name=f"{layer_name}_drop")
                )
            else:
                self.hidden_dropout.append(None)
        self.output_layer = Dense(
            num_classes, activation=output_activation, name=f"{prefix}_output"
        )

    def call(self, inputs, training=None):
        Z = inputs
        for layer, activation, batchnorm, dropout in zip(
            self.hidden,
            self.hidden_activations,
            self.hidden_batchnorm,
            self.hidden_dropout,
        ):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z, training=training)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z, training=training)
        out = self.output_layer(Z)
        return out


class Autoencoder(keras.Model):
    def __init__(
        self,
        ae_hidden_size,
        ae_hidden_dropout=None,
        ae_activation="relu",
        ae_bottleneck_activation="linear",
        ae_output_activation="relu",
        ae_init="glorot_uniform",
        ae_batchnorm=True,
        ae_l1_enc_coef=0,
        ae_l2_enc_coef=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        center_idx = int(
            np.floor(len(ae_hidden_size) / 2.0)
        )  # index of the bottleneck layer
        if not ae_hidden_dropout:
            ae_hidden_dropout = [0] * len(ae_hidden_size)
        if type(ae_hidden_dropout) == float:
            ae_hidden_dropout = [ae_hidden_dropout] * len(ae_hidden_size)
        if len(ae_hidden_dropout) == 1:
            ae_hidden_dropout = ae_hidden_dropout * len(ae_hidden_size)
        self.enc_hidden_size = ae_hidden_size[: center_idx + 1]
        self.dec_hidden_size = ae_hidden_size[center_idx + 1 :]
        self.enc_hidden_dropout = ae_hidden_dropout[: center_idx + 1]
        self.dec_hidden_dropout = ae_hidden_dropout[center_idx + 1 :]
        self.ae_bottleneck_activation = ae_bottleneck_activation
        self.enc = Encoder(
            self.enc_hidden_size,
            hidden_dropout=self.enc_hidden_dropout,
            activation=ae_activation,
            bottleneck_activation=ae_bottleneck_activation,
            init=ae_init,
            batchnorm=ae_batchnorm,
            l1_enc_coef=ae_l1_enc_coef,
            l2_enc_coef=ae_l2_enc_coef,
            name="Encoder",
        )
        self.dec = Decoder(
            self.dec_hidden_size,
            hidden_dropout=self.dec_hidden_dropout,
            activation=ae_activation,
            init=ae_init,
            batchnorm=ae_batchnorm,
            name="Decoder",
        )
        self.ae_output_activation = ae_output_activation

    def build(self, batch_input_size):
        if type(batch_input_size) == dict:
            count_input_size = batch_input_size["counts"]
        else:
            count_input_size = batch_input_size
        self.ae_output_layer = Dense(
            count_input_size[-1],
            activation=self.ae_output_activation,
            name="autoencoder_output",
        )
        super().build(batch_input_size)

    def call(self, inputs, training=None):
        if (
            type(inputs) != dict
        ):  # In this case, inputs is only the count matrix
            Z = inputs
            sf_layer = np.array([1.0] * inputs.shape[-1]).astype(
                float
            )  # TODO : this doesn't work currently
        else:
            Z = inputs["counts"]
            sf_layer = inputs["size_factors"]
        enc_layer = self.enc(Z, training=training)
        dec_layer = self.dec(enc_layer, training=training)
        mean = self.ae_output_layer(dec_layer)
        out = ColwiseMultLayer([mean, sf_layer])
        return {"bottleneck": enc_layer, "reconstruction": out}


def predict_embedding(self, inputs):
    return self.enc(inputs)


class Classif_Autoencoder(Autoencoder):
    def __init__(
        self,
        num_classes,
        class_hidden_size,
        class_hidden_dropout=None,
        class_batchnorm=True,
        class_activation="relu",
        class_output_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not class_hidden_dropout:
            class_hidden_dropout = [0] * len(class_hidden_size)
        if type(class_hidden_dropout) == float:
            class_hidden_dropout = [class_hidden_dropout] * len(
                class_hidden_size
            )
        if len(class_hidden_dropout) == 1:
            class_hidden_dropout = class_hidden_dropout * len(
                class_hidden_size
            )
        self.classifier = Classifier(
            num_classes=num_classes,
            hidden_size=class_hidden_size,
            prefix="classifier",  #
            hidden_dropout=class_hidden_dropout,
            batchnorm=class_batchnorm,
            activation=class_activation,
            output_activation=class_output_activation,
            name="Classifier",
        )

    def call(self, inputs, training=None):
        if (
            type(inputs) != dict
        ):  # In this case, inputs is only the count matrix
            Z = inputs
            sf_layer = np.array([1.0] * inputs.shape[-1]).astype(
                float
            )  # TODO : this doesn't work currently
        else:
            Z = inputs["counts"]
            sf_layer = inputs["size_factors"]
        enc_layer = self.enc(Z, training=training)
        dec_layer = self.dec(enc_layer, training=training)
        clas_out = self.classifier(enc_layer, training=training)
        mean = self.ae_output_layer(dec_layer)
        out = ColwiseMultLayer([mean, sf_layer])
        return {
            "bottleneck": enc_layer,
            "classifier": clas_out,
            "reconstruction": out,
        }


class DANN_AE(Classif_Autoencoder):
    def __init__(
        self,
        num_batches,
        dann_hidden_size,
        dann_hidden_dropout=None,
        dann_batchnorm=True,
        dann_activation="relu",
        dann_output_activation="softmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not dann_hidden_dropout:
            dann_hidden_dropout = [0] * len(dann_hidden_size)
        if type(dann_hidden_dropout) == float:
            dann_hidden_dropout = [dann_hidden_dropout] * len(dann_hidden_size)
        if len(dann_hidden_dropout) == 1:
            dann_hidden_dropout = dann_hidden_dropout * len(dann_hidden_size)
        self.dann_discriminator = Classifier(
            num_classes=num_batches,
            hidden_size=dann_hidden_size,
            prefix="dann_discriminator",
            hidden_dropout=dann_hidden_dropout,
            batchnorm=dann_batchnorm,
            activation=dann_activation,
            output_activation=dann_output_activation,
            name="Dann_Discriminator",
        )
        self.grad_reverse = GradReverse()

    def call(self, inputs, training=None):
        if (
            type(inputs) != dict
        ):  # In this case, inputs is only the count matrix
            Z = inputs
            sf_layer = tf.ones(
                inputs.shape
            )  # TODO : this doesn't work currently
        else:
            Z = inputs["counts"]
            sf_layer = inputs["size_factors"]
        enc_layer = self.enc(Z, training=training)
        dec_layer = self.dec(enc_layer, training=training)
        clas_out = self.classifier(enc_layer, training=training)
        dann_out = self.dann_discriminator(
            self.grad_reverse(enc_layer), training=training
        )
        mean = self.ae_output_layer(dec_layer)
        out = ColwiseMultLayer([mean, sf_layer])
        return {
            "bottleneck": enc_layer,
            "classifier": clas_out,
            "batch_discriminator": dann_out,
            "reconstruction": out,
        }

    def build_graph(
        ae, input_shape
    ):  # TODO : Make sure that this doesn't block the model
        """
        ex : build_graph(dann_ae, {'counts': (32,dataset.n_vars, ), "size_factors":(32,1,)})
        """
        input_shape_nobatch = input_shape["counts"][1:]
        input_shape_nobatch_sf = input_shape["size_factors"][1:]
        ae.build(input_shape)
        inputs = {
            "counts": tf.keras.Input(input_shape_nobatch),
            "size_factors": tf.keras.Input(input_shape_nobatch_sf),
        }

        if not hasattr(ae, "call"):
            raise AttributeError(
                "User should define 'call' method in sub-class model!"
            )

        _ = ae.call(inputs)
