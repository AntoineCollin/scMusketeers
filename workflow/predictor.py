try :
    import keras
    from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda
    from keras.models import Model
    import tensorflow as tf
    from keras.utils import np_utils
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, plot_confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    pass

import anndata
import pandas as pd
import numpy as np
import os
import scanpy as sc
import anndata
import yaml
import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_one_hot(adata, obs_key):
    X = adata.X
    if type(X) != np.ndarray:
        X_array = X.toarray()
    else:
        X_array=X

    y = adata.obs[obs_key]
    y_array = y.array
    
    # encode class values as integers
    encoder = LabelEncoder()  
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    decoded_y = encoder.inverse_transform(encoded_y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_y)

    X_train, X_test, y_train, y_test = train_test_split(X_array, dummy_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoder
    
def build_predictor(input_size, nb_classes):
    inputs = Input(shape=(input_size,))

    # a layer instance is callable on a tensor, and returns a tensor
    denses = Dense(128, activation='relu')(inputs)
    denses = Dense(64, activation='relu')(denses)
    predictions = Dense(nb_classes, activation='softmax')(denses)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.summary()
    return model

def batch_generator_training(X, y, batch_size):
    samples_per_epoch = X.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        X_batch = np.array(X_batch).reshape(X_batch.shape[0], X_batch.shape[1], 1)
        y_batch = y[index_batch,:]
        counter += 1
        yield(np.array(X_batch),np.array(y_batch))
        if (counter >= number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

class MLP_Predictor:
    def __init__(self,latent_space, predict_key, predictor_hidden_sizes, predictor_epochs, predictor_batch_size, unlabeled_category, predictor_activation = 'softmax'): #random_state gives the split for train_test split during the MLP predictor training
        self.predict_key = predict_key
        self.adata = latent_space
        self.predictor_hidden_sizes = predictor_hidden_sizes
        self.predictor_epochs = predictor_epochs
        self.predictor_batch_size = predictor_batch_size
        self.predictor_activation = predictor_activation
        self.unlabeled_category = unlabeled_category
        
        ## Removing the unlabeled cells from the prediction training set
        to_keep = self.adata.obs['train_split'].copy()
        UNK_cells = self.adata.obs[self.predict_key] == self.unlabeled_category
        print('nb of nan')
        print(UNK_cells.sum())
        to_keep[UNK_cells] = 'val'
        self.adata.obs['train_split'] = to_keep
        
        self.adata_train = self.adata[self.adata.obs['train_split'] == 'train', : ].copy()
        self.adata_test = self.adata[self.adata.obs['train_split'] == 'test', : ].copy()
        self.adata_val = self.adata[self.adata.obs['train_split'] == 'val', : ].copy()

        self.train_index = self.adata_train.obs.index
        self.test_index = self.adata_test.obs.index
        self.val_index = self.adata_val.obs.index

        self.y = self.adata.obs[self.predict_key]
        self.categories = self.y.unique()
        self.train_categories = []
        self.model = keras.Model()
        self.X_array = np.array([])
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        self.y_train_onehot = np.array([])
        self.y_test_onehot = np.array([])
        self.y_pred_raw = np.array([]) # The output of the model ie result of the softmax/sigmoid layer
        self.y_pred = pd.Series()
        self.prediction_table = pd.DataFrame()
        self.encoder = LabelEncoder()
        self.train_adata = anndata.AnnData()
        self.test_adata = anndata.AnnData()
        self.val_adata = anndata.AnnData()
        self.train_history = keras.callbacks.History()
        self.history = dict()
        self.is_trained = False
        self.training_time = datetime.datetime.today()
        
    def preprocess_one_hot(self):
        self.X_array = self.adata.X
        if type(self.X_array) != np.ndarray:
            self.X_array = self.X_array.toarray()
        
        self.X_train = self.adata_train.X
        if type(self.X_train) != np.ndarray:
            self.X_train = self.X_train.toarray()
        
        self.X_test = self.adata_test.X
        if type(self.X_test) != np.ndarray:
            self.X_test = self.X_test.toarray()
        
        self.X_val = self.adata_val.X
        if type(self.X_val) != np.ndarray:
            self.X_val = self.X_val.toarray()
        
        self.y_train = self.y[self.train_index]
        self.y_test = self.y[self.test_index]
        self.y_val = self.y[self.val_index]
        self.train_categories = self.y_train.unique()
        
        # encode class values as integers
        self.encoder = LabelEncoder()  
        self.encoder.fit(self.y_train)
        self.y_train_onehot = self.encoder.transform(self.y_train)
#         self.y_test_onehot = self.encoder.transform(self.y_test)

        self.y_train_onehot = np_utils.to_categorical(self.y_train_onehot)
#         self.y_test_onehot = np_utils.to_categorical(self.y_test_onehot)
        
        self.train_adata = self.adata[self.train_index,:]
        self.test_adata = self.adata[self.test_index,:]
        self.val_adata = self.adata[self.val_index,:]
        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_val, self.y_test, self.encoder

    def build_predictor(self):
        print('building predictor on :')
        print(self.adata_train)
        print(self.X_train)
        print(self.categories)
        print(self.train_categories)
        print(self.adata_train.obs[self.predict_key].value_counts())

        input_size = self.X_train.shape[1]
        nb_classes = len(self.train_categories)
        
        inputs = Input(shape=(input_size,))

        denses = Dense(self.predictor_hidden_sizes, activation='relu')(inputs)
        predictions = Dense(nb_classes, activation=self.predictor_activation)(denses)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        model.summary()
        self.model = model
        return model
     
    def train_model(self):
        self.training_time = datetime.datetime.today()
        batch_size = self.predictor_batch_size
        epochs = self.predictor_epochs
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
        callbacks = [es]

        self.train_history = self.model.fit(x=self.X_train, y=self.y_train_onehot,
                                            batch_size=batch_size,epochs = epochs, callbacks=callbacks, 
                                            validation_split=0.2)
        self.is_trained = True
        
    def predict_on_test (self):
        self.y_pred_raw = self.model.predict(self.X_array)
        self.y_pred = pd.Series(self.encoder.inverse_transform(np.argmax(self.y_pred_raw, axis = 1)), index = self.y.index)
        self.prediction_table = pd.DataFrame({'y_true' :self.y, 'y_pred':self.y_pred, 'is_test':self.y.index.isin(self.test_index)}, index=self.y.index)
        self.adata.obs['y_pred'] = self.y_pred
        self.adata.obsm['y_pred_raw'] = self.y_pred_raw
        self.adata.uns['prediction_decoder'] = self.encoder.classes_
                                                
    def save_model(self, save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        self.model.save(save_path)

    def save_results(self):
        self.prediction_table.to_csv(self.predict_save_path)
        if not os.path.isdir(self.model_predict_save_path):
            os.mkdir(self.model_predict_save_path)
        self.model.save(self.model_predict_save_path + '/model')
        with open(self.model_predict_save_path + '/history', 'wb') as file_pi:
            pickle.dump(self.train_history.history, file_pi)
            
    def load_prediction_results(self):
        self.prediction_table = pd.read_csv(self.predict_save_path, index_col=0)
        self.history = pickle.load(open(self.model_predict_save_path + '/history', "rb"))
        self.adata.obs[f'{self.predict_key}_predictions'] = self.prediction_table['y_pred']
        misclassified = self.adata.obs[self.predict_key].astype('str')
        misclassified[self.adata.obs[f'{self.predict_key}_predictions'] != self.adata.obs[self.predict_key]] = 'misclassified'
        self.adata.obs[f'misclassified'] = misclassified
        misclassified_and_test = self.adata.obs['misclassified'].astype('str')
        misclassified_and_test[(self.adata.obs['misclassified'] == 'misclassified') & (self.prediction_table['is_test'])] = 'misclassified_and_test'
        self.adata.obs[f'misclassified_and_test'] = misclassified_and_test
        
        self.adata.obs[f'is_test'] = self.prediction_table['is_test'].replace({True:'test', False:'train'})
        
    def plot_prediction_results(self, test_only, normalize = 'true',ax=None):
        if not test_only:
            y_true = self.prediction_table['y_true']
            y_pred = self.prediction_table['y_pred']
        else:
            y_true = self.prediction_table.loc[self.prediction_table['is_test'],'y_true']
            y_pred = self.prediction_table.loc[self.prediction_table['is_test'],'y_pred']
        labels = sorted(y_true.unique())
        self.confusion_matrix = confusion_matrix(y_true, y_pred, labels = labels, normalize=normalize)
        confusion_to_plot=pd.DataFrame(self.confusion_matrix, index = labels, columns=labels)
        self.balanced_accuracy_score = balanced_accuracy_score(y_true, y_pred)
        self.accuracy_score = accuracy_score(y_true, y_pred)
        plt.figure(figsize = (15,15))
        ax = sns.heatmap(confusion_to_plot, annot=True, ax=ax)
        return self.confusion_matrix, self.balanced_accuracy_score, self.accuracy_score, ax
    
    def plot_training_performances(self):
        plt.figure(figsize = (5,5))
        plt.show()
        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
    def get_colors(self, color=None):
        if not color:
            color = self.predict_key
        if not self.adatas:
            print('please load at least one dataset')
        else : 
            adata = self.adatas[0]
            sc.pl.umap(adata, color = color, show=False)
            self.colors[color] = adata.uns[f'{color}_colors']
            self.colors_UNK[f'{color} + _UNK'] = self.colors[color] + ['#FF0000']
            
    def plot_misclassified_umap(self, title = None):
        if self.prediction_table.empty:
            self.load_prediction_results()
        sc.pl.umap(self.adata, color = self.predict_key, title=title)
        self.adata.uns[f'misclassified_colors'] = self.adata.uns[f'{self.predict_key}_colors'] + ['#FF0000']
        self.adata.uns[f'misclassified_and_test_colors'] = self.adata.uns[f'{self.predict_key}_colors'] + ['#FF0000'] + ['#08ff00']
        sc.pl.umap(self.adata, color = 'misclassified', title=title)
        sizes = pd.Series([2]* self.adata.n_obs, index = self.adata.obs.index)
        sizes[self.adata.obs[f'{self.predict_key}_predictions'] != self.adata.obs[self.predict_key]] = 20
        sc.pl.umap(self.adata, color = [self.predict_key, f'{self.predict_key}_predictions'], size=sizes)
        sc.pl.umap(self.adata, color = ['misclassified','is_test'], size=sizes)
        sc.pl.umap(self.adata, color = ['misclassified_and_test', 'dca_split'], size=sizes)

       
        