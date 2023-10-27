import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda, Layer
from keras.models import Model
from keras.regularizers import l1_l2
import tensorflow as tf
import numpy as np
from .layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer, ElementwiseDense,GradReverse

advanced_activations = ('PReLU', 'LeakyReLU')

class Encoder(keras.Model):
    def __init__(self, hidden_size,
                    hidden_dropout=None,
                    activation='relu',
                    init='glorot_uniform',
                    batchnorm=False,
                    l1_enc_coef=0,
                    l2_enc_coef=0,
                    **kwargs):
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0]*len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []
        center_idx = len(hidden_size)

        for i, (hid_size, hid_drop) in enumerate(zip(hidden_size, hidden_dropout)):
            if i != center_idx:
                layer_name = f'enc_{str(i)}'
                stage = 'encoder'
            else :
                layer_name = f'center'
                stage = 'center'

            self.hidden.append(Dense(hid_size, activation=None, kernel_initializer=init,
                kernel_regularizer=l1_l2(l1_enc_coef, l2_enc_coef),
                name=layer_name))
            
            if batchnorm:
                self.hidden_batchnorm.append(BatchNormalization(center=True, scale=False))
            else : 
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if activation in advanced_activations:
                self.hidden_activations.append(keras.layers.__dict__[activation](name=f'{layer_name}_act'))
            else:
                self.hidden_activations.append(Activation(activation, name=f'{layer_name}_act'))

            if hid_drop > 0.0:
                self.hidden_dropout.append(Dropout(hid_drop, name=f'{layer_name}_drop'))
            else:
                self.hidden_dropout.append(None)

    def call(self, inputs):
        Z = inputs 
        for layer, activation, batchnorm, dropout in zip(self.hidden,self.hidden_activations, self.hidden_batchnorm, self.hidden_dropout):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z)           
        return Z

    def build_graph(self, dim):
        x = Input(shape=(dim))
        return Model(inputs=[x], outputs=self.call(x))

class Decoder(keras.layers.Layer):
    def __init__(self, hidden_size,
                    hidden_dropout,
                    activation='relu',
                    init='glorot_uniform',
                    batchnorm=False,
                    **kwargs):
        '''
        hidden_size : A list of the size for the decoder only. Ex : if the size of the total AE are [200,100,10,100,200], the hidden_size 
        for the Decoder should be [100,200]
        '''
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0]*len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []

        for i, (hid_size, hid_drop) in enumerate(zip(hidden_size, hidden_dropout)):
            layer_name = f'dec_{str(i)}'
            stage = 'decoder'
            self.hidden.append(Dense(hid_size, activation=None, kernel_initializer=init, name=layer_name))
            
            if batchnorm:
                self.hidden_batchnorm.append(BatchNormalization(center=True, scale=False))
            else : 
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if activation in advanced_activations:
                self.hidden_activations.append(keras.layers.__dict__[activation](name=f'{layer_name}_act'))
            else:
                self.hidden_activations.append(Activation(activation, name=f'{layer_name}_act'))

            if hid_drop > 0.0:
                self.hidden_dropout.append(Dropout(hid_drop, name=f'{layer_name}_drop'))
            else:
                self.hidden_dropout.append(None)

    def call(self, inputs):
        Z = inputs 
        for layer, activation, batchnorm, dropout in zip(self.hidden,self.hidden_activations, self.hidden_batchnorm, self.hidden_dropout):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z)           
        return Z

    # def build_graph(self, dim):
    #     x = Input(shape=(dim))
    #     return Model(inputs=[x], outputs=self.call(x))

class Classifier(keras.Model):
    def __init__(self, num_classes,
                hidden_size,
                name = 'classifier',
                hidden_dropout=None,
                batchnorm=False,
                activation='relu',
                output_activation='softmax',
                **kwargs):
        super().__init__(**kwargs)
        if not hidden_dropout:
            hidden_dropout = [0]*len(hidden_size)
        self.hidden = []
        self.hidden_activations = []
        self.hidden_batchnorm = []
        self.hidden_dropout = []
        for i, (hid_size, hid_drop) in enumerate(zip(hidden_size, hidden_dropout)):
            layer_name = f'{name}_{str(i)}'
            self.hidden.append(Dense(hid_size, activation=None, name=layer_name))

            if batchnorm:
                self.hidden_batchnorm.append(BatchNormalization(center=True, scale=False))
            else : 
                self.hidden_batchnorm.append(None)
            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if activation in advanced_activations:
                self.hidden_activations.append(keras.layers.__dict__[activation](name=f'{layer_name}_act'))
            else:
                self.hidden_activations.append(Activation(activation, name=f'{layer_name}_act'))

            if hid_drop > 0.0:
                self.hidden_dropout.append(Dropout(hid_drop, name=f'{layer_name}_drop'))
            else:
                self.hidden_dropout.append(None)
        self.output_layer = Dense(num_classes, activation = output_activation, name = f'{name}_output')

    def call(self,inputs):
        Z = inputs 
        for layer, activation, batchnorm, dropout in zip(self.hidden,self.hidden_activations, self.hidden_batchnorm, self.hidden_dropout):
            Z = layer(Z)
            if batchnorm:
                Z = batchnorm(Z)
            if activation:
                Z = activation(Z)
            if dropout:
                Z = dropout(Z)
        out = self.output_layer(Z)
        return out


    class Autoencoder(keras.Model):
        def __init__(self, 
                    ae_hidden_size,
                    ae_hidden_dropout=None,
                    ae_activation='relu',
                    ae_output_activation='linear',
                    ae_init='glorot_uniform',
                    ae_batchnorm=False,
                    ae_l1_enc_coef=0,
                    ae_l2_enc_coef=0,
                    **kwargs ):
            super().__init__(**kwargs)
            center_idx = int(np.floor(len(ae_hidden_size) / 2.0)) # index of the bottleneck layer
            if not ae_hidden_dropout:
                ae_hidden_dropout = [0]*len(ae_hidden_size)
            self.enc_hidden_size = ae_hidden_size[:center_idx+1]
            self.dec_hidden_size = ae_hidden_size[center_idx+1:]
            self.enc_hidden_dropout = ae_hidden_dropout[:center_idx+1]
            self.dec_hidden_dropout = ae_hidden_dropout[center_idx+1:]
            self.enc = Encoder(self.enc_hidden_size,
                    hidden_dropout = self.enc_hidden_dropout,
                    activation=ae_activation,
                    init=ae_init,
                    batchnorm = ae_batchnorm,
                    l1_enc_coef=ae_l1_enc_coef,
                    l2_enc_coef=ae_l2_enc_coef)
            self.dec = Decoder(self.dec_hidden_size,
                            hidden_dropout=self.dec_hidden_dropout,
                            activation=ae_activation,
                            init=ae_init,
                            batchnorm = ae_batchnorm)
            self.ae_output_activation = ae_output_activation

        def build(self, batch_input_size):
            if type(batch_input_size) == dict:
                count_input_size = batch_input_size['counts']
            else : 
                count_input_size = batch_input_size
            self.ae_output_layer = Dense(count_input_size[-1], activation=self.ae_output_activation, name = 'autoencoder_output')
            super().build(batch_input_size)

        def call(self, inputs):
            if type(inputs) != dict: # In this case, inputs is only the count matrix
                Z = inputs
                sf_layer = np.array([1.]*inputs.shape[-1]).astype(float) # TODO : this doesn't work currently
            else:
                Z = inputs['counts']
                sf_layer = inputs['size_factors']
            enc_layer = self.enc(Z)
            dec_layer = self.dec(enc_layer)
            mean = self.ae_output_layer(dec_layer)
            out = ColwiseMultLayer([mean, sf_layer])
            return {'bottleneck':enc_layer,'reconstruction': out}
    
    def predict_embedding(self,inputs): 
        return self.enc(inputs)
    
    
class Classif_Autoencoder(Autoencoder):
    def __init__(self, 
                 num_classes,
                 class_hidden_size,
                 class_hidden_dropout=None,
                 class_batchnorm=False,
                 class_activation='relu',
                 class_output_activation='softmax', **kwargs):
        super().__init__(**kwargs)
        if not class_hidden_dropout:
            class_hidden_dropout = [0]*len(class_hidden_size)
        self.classifier = Classifier(num_classes=num_classes,
                                     hidden_size=class_hidden_size,
                                     name = 'classifier',
                                     hidden_dropout=class_hidden_dropout,
                                     batchnorm=class_batchnorm,
                                     activation=class_activation,
                                     output_activation=class_output_activation)

    def call(self, inputs):
        if type(inputs) != dict: # In this case, inputs is only the count matrix
            Z = inputs
            sf_layer = np.array([1.]*inputs.shape[-1]).astype(float) # TODO : this doesn't work currently
        else:
            Z = inputs['counts']
            sf_layer = inputs['size_factors']
        enc_layer = self.enc(Z)
        dec_layer = self.dec(enc_layer)
        clas_out = self.classifier(enc_layer)
        mean = self.ae_output_layer(dec_layer)
        out = ColwiseMultLayer([mean, sf_layer])
        return {'bottleneck':enc_layer,'classifier': clas_out,'reconstruction': out}
    
class DANN_AE(Classif_Autoencoder):
    def __init__(self, 
                 num_batches,
                 dann_hidden_size,
                 dann_hidden_dropout=None,
                 dann_batchnorm=False,
                 dann_activation='relu',
                 dann_output_activation='softmax', **kwargs):
        super().__init__(**kwargs)
        if not dann_hidden_dropout:
            dann_hidden_dropout = [0]*len(dann_hidden_size)
        self.dann_discriminator = Classifier(num_classes=num_batches,
                                             hidden_size=dann_hidden_size,
                                             name = 'dann_discriminator',
                                             hidden_dropout=dann_hidden_dropout,
                                             batchnorm=dann_batchnorm,
                                             activation=dann_activation,
                                             output_activation=dann_output_activation)
        self.grad_reverse = GradReverse()

    def call(self, inputs):
        if type(inputs) != dict: # In this case, inputs is only the count matrix
            Z = inputs
            sf_layer = tf.ones(inputs.shape) # TODO : this doesn't work currently
        else:
            Z = inputs['counts']
            sf_layer = inputs['size_factors']
        enc_layer = self.enc(Z)
        dec_layer = self.dec(enc_layer)
        clas_out = self.classifier(enc_layer)
        dann_out = self.grad_reverse(self.dann_discriminator(enc_layer))
        mean = self.ae_output_layer(dec_layer)
        out = ColwiseMultLayer([mean, sf_layer])
        return {'bottleneck':enc_layer,'classifier': clas_out,'batch_discriminator':dann_out,'reconstruction': out}