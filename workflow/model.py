import anndata
import scanpy as sc
import sys
import os
import numpy as np


try :
    import pandas as pd
    import scarches as sca
    import urllib.request 
    import gzip
    import shutil

    import sys
    sys.path.append("/home/acollin/dca_permuted_workflow/LCA_scarches_reference/scripts")
    import scarches_label_transfer
    import data_import_and_cleaning
except ImportError:
    pass

try :
    import torch
except ImportError:
    pass
try:
    import scvi
except ImportError:
    pass

sys.path.insert(1, os.path.join(sys.path[0], '..'))
try :
    import tensorflow.compat.v1 as tf
except ImportError:
    pass

try :
    from dca.io import read_dataset, normalize, train_split, make_unk
except ImportError:
    pass

try :
    from dca.train import train
except ImportError:
    pass

try :
    from dca.network import AE_types
except ImportError:
    pass

try :
    from dca.paths import results_save_path,dataset_path
except ImportError:
    pass
# try : 
#     from tensorflow.python.keras.backend import set_session, get_session
# except ImportError:
#     pass

    
class DCA_Permuted:
    def __init__(self, ae_type='zinb-conddisp',hidden_size=(64, 32, 64), hidden_dropout=0, batchnorm=True, activation='relu', init='glorot_uniform',
                 epochs=10,reduce_lr=10, early_stop=5, batch_size=32, optimizer='adam',verbose=True,threads=None, learning_rate=None,
                 training_kwds={},network_kwds={}, n_perm=1, class_key='celltype', permute = True,change_perm=True, unlabeled_category = 'UNK',use_raw_as_output=False,
                 contrastive_margin = 0, same_class_pct = None):
        self.net_name = 'dca_permuted'
        self.net = None
        self.ae_type = ae_type
        self.class_key = class_key
        self.network_kwds = network_kwds
        self.network_kwds = {**self.network_kwds,
                             'hidden_size' : hidden_size,
                             'hidden_dropout' : hidden_dropout,
                             'batchnorm' : batchnorm,
                             'activation' : activation,
                             'init' : init}
        if self.ae_type == 'contrastive':
            self.network_kwds['contrastive_margin'] = contrastive_margin
        self.training_kwds = training_kwds
        self.training_kwds = {**self.training_kwds,
                              'epochs': epochs,
                              'reduce_lr': reduce_lr,
                              'early_stop': early_stop,
                              'batch_size': batch_size,
                              'optimizer': optimizer,
                              'verbose': verbose,
                              'threads': threads,
                              'learning_rate': learning_rate,
                              'permute': permute,
                              'change_perm':change_perm,
                              'class_key': class_key,
                              'n_perm': n_perm,
                              'same_class_pct': same_class_pct}
        self.unlabeled_category = unlabeled_category
        self.hist = dict()
        self.use_raw_as_output = use_raw_as_output
        
    def make_net(self, Dataset):
        input_size = output_size = Dataset.adata_train.n_vars
        self.net = AE_types[self.ae_type](input_size=input_size,
                                output_size=output_size,
                                **self.network_kwds)
        #net.save()
        self.net.build()
      
    def train_net(self, Dataset):
        if type(Dataset.adata_train.X) != np.ndarray :
            Dataset.adata_train.X = Dataset.adata_train.X.toarray()
        self.hist = train(adata = Dataset.adata_train, network = self.net, use_raw_as_output=self.use_raw_as_output,unlabeled_category = self.unlabeled_category, **self.training_kwds)
        return self.hist 
        
    def predict_net(self, Dataset):
        print('model.predict : ')
#         print('X with net.predict (supposedly corrected counts is :)')
#         print(aaa.X)
#         print('dispersion :')
#         print(aaa.obsm['X_dca_dispersion'])
#         print(aaa.obsm['X_dca_dispersion'].shape)
#         print('dropout :')
#         print(aaa.obsm['X_dca_dropout'])
#         print('dispersion :')
#         print(self.net.extra_models['dispersion'])
#         print('dropout :')
#         print(self.net.extra_models['pi'].predict(Dataset.adata.X))
        
        if self.ae_type == 'contrastive':
            corrected_count = anndata.AnnData() # Does not yet handle returning corrected counts
            latent_space = anndata.AnnData(self.net.encoder.predict({'count': Dataset.adata.X, 'size_factors': Dataset.adata.obs.size_factors, 'similarity':np.array([True] * Dataset.adata.n_obs)}), obs=Dataset.adata.obs)
        # similarity is just a place holder here it only has an impact during training anyway
        else :
            corrected_count= self.net.predict(Dataset.adata, mode='denoise', return_info=True, copy=True)
            latent_space = anndata.AnnData(self.net.encoder.predict({'count': Dataset.adata.X, 'size_factors': Dataset.adata.obs.size_factors}), obs=Dataset.adata.obs)
        return latent_space,corrected_count
    
    def save_net(self, save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path) 
        self.net.model.save(save_path, overwrite=True)

class DCA_into_Perm:
    """
    Training in three steps :
    - First denoise data using DCA
    - Train a normal AE in regular fashion to ease optimization on the denoised data
    - Train the same AE with permutation to finish training
    This should help with the optimization step, at the moment, training with permutation doesn't really converge.
    """
    def __init__(self, ae_type='zinb-conddisp',hidden_size=(64, 32, 64), hidden_dropout=0, batchnorm=True, activation='relu', init='glorot_uniform',
                 epochs=10,reduce_lr=10, early_stop=5, batch_size=32, optimizer='adam',verbose=True,threads=None, learning_rate=None,
                 training_kwds={},network_kwds={}, n_perm=1, change_perm=True,class_key='celltype', unlabeled_category = 'UNK',use_raw_as_output=False):
        self.denoise_net = None
        self.permute_net = None
        self.ae_type = ae_type
        self.class_key = class_key
        self.network_kwds = network_kwds
        self.network_kwds = {**self.network_kwds,
                             'hidden_size' : hidden_size,
                             'hidden_dropout' : hidden_dropout,
                             'batchnorm' : batchnorm,
                             'activation' : activation,
                             'init' : init}
        self.training_kwds = training_kwds
        self.training_kwds = {**self.training_kwds,
                              'epochs': epochs,
                              'reduce_lr': reduce_lr,
                              'early_stop': early_stop,
                              'batch_size': batch_size,
                              'optimizer': optimizer,
                              'verbose': verbose,
                              'threads': threads,
                              'change_perm': change_perm,
                              'learning_rate': learning_rate,
                              'class_key': class_key,
                              'n_perm': n_perm}
        self.unlabeled_category = unlabeled_category
        self.hist_denoising = dict()
        self.hist_dimred = dict()
        self.use_raw_as_output = use_raw_as_output
        self.denoised_full_adata = anndata.AnnData()

    def make_net(self, Dataset):
        input_size = output_size = Dataset.adata_train.n_vars
#         global session_denoise
#         global session_permute
#         session_denoise = tf.Session()
#         session_permute = tf.Session()
#         set_session(session_denoise)
        self.denoise_net = AE_types[self.ae_type](input_size=input_size,
                                output_size=output_size,
                                **self.network_kwds)
        self.permute_net = AE_types['normal'](input_size=input_size,
                                output_size=output_size,
                                **self.network_kwds)
        #net.save()
        print(self.network_kwds)
#         self.denoise_net.build()
#         self.permute_net.build()

    def train_net(self, Dataset):
        if type(Dataset.adata_train.X) != np.ndarray :
            Dataset.adata_train.X = Dataset.adata_train.X.toarray()
        Dataset.adata_train.raw = Dataset.adata_train
        self.denoise_net.build()
        self.hist_denoising = train(adata = Dataset.adata_train, network = self.denoise_net, use_raw_as_output=self.use_raw_as_output, permute = False,
                                    unlabeled_category = self.unlabeled_category, **self.training_kwds) # Here unlabeled_category isn't considered since permute = False
        Dataset.adata_train.X = Dataset.adata_train.raw.X # Making sure .X preserves raw values after predict (which can modify the .X in place)
        denoised_train_adata = self.denoise_net.predict(Dataset.adata_train, mode='denoise', return_info=True, copy=True)
        Dataset.adata.raw = Dataset.adata
        self.denoised_full_adata = self.denoise_net.predict(Dataset.adata, mode='denoise', return_info=True, copy=True) # Computing denoised adata to prevent later initialization errors
        Dataset.adata.X = Dataset.adata.raw.X # Making sure .X preserves raw values after predict (which can modify the .X in place)
        denoised_train_adata.obs['size_factors'] = denoised_train_adata.obs.n_counts / np.median(denoised_train_adata.obs.n_counts)
        self.permute_net.build()
        self.hist_dimred = train(adata = denoised_train_adata, network = self.permute_net, use_raw_as_output=self.use_raw_as_output, permute = True,
                                 unlabeled_category = self.unlabeled_category, **self.training_kwds)
        return self.hist_dimred
    
    def save_net(self, save_path):
        denoiser_save_path = save_path + 'denoiser/'
        dim_red_save_path = save_path + 'dim_red/'
        if not os.path.isdir(save_path):
            os.makedirs(denoiser_save_path)
            os.makedirs(dim_red_save_path)
#         self.denoise_net.model.save(denoiser_save_path, overwrite=True)
#         self.permute_net.model.save(dim_red_save_path, overwrite=True)
        
    def predict_net(self, Dataset):
        Dataset.adata.raw = Dataset.adata
        # denoised_adata = self.denoise_net.predict(Dataset.adata, mode='denoise', return_info=True, copy=True)
        # Dataset.adata.X = Dataset.adata.raw.X # Making sure .X preserves raw values after predict (which can modify the .X in place)
        self.denoised_full_adata.obs['size_factors'] = self.denoised_full_adata.obs.n_counts / np.median(self.denoised_full_adata.obs.n_counts)
        latent_space = anndata.AnnData(self.permute_net.encoder.predict({'count': self.denoised_full_adata.X, 'size_factors': self.denoised_full_adata.obs.size_factors}), obs=Dataset.adata.obs)
        corrected_count= self.permute_net.predict(self.denoised_full_adata, mode='denoise', return_info=True, copy=True)
        return latent_space,corrected_count
        
        
class Scanvi:
    def __init__(self, unlabeled_category = 'UNK', class_key='celltype',n_samples_per_label =None, n_latent=50, gene_likelihood='zinb'):
        self.net = None
        self.vae= None
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.class_key = class_key
        self.unlabeled_category = unlabeled_category
        self.n_samples_per_label = n_samples_per_label
        
    def make_net(self, Dataset):
        adata_train = Dataset.adata_train
        print('entering scvi')
        print('training set is ')
        print(adata_train)
        for b_k in ['manip', 'sample']:
            if b_k in adata_train.obs.columns:
                batch_key = b_k
        scvi.model.SCVI.setup_anndata(adata_train, labels_key=self.class_key, batch_key=batch_key)
        print(adata_train)
        print(adata_train.obs[self.class_key])
        print(adata_train.X[:100,:100])
        self.vae = scvi.model.SCVI(adata_train, n_layers=2, n_latent=self.n_latent, gene_likelihood=self.gene_likelihood)
        print(self.vae)
        print(scvi.data.view_anndata_setup(self.vae.adata))

      
    def train_net(self, Dataset):
        print('starting train')
        self.vae.train()
        self.net = scvi.model.SCANVI.from_scvi_model(
            self.vae,
            adata=Dataset.adata_train,
            unlabeled_category=self.unlabeled_category,
        )

        self.net.train(max_epochs=20, n_samples_per_label=self.n_samples_per_label)
        self.hist = self.net.history
        return self.hist 
        
    def predict_net(self, Dataset):
        corrected_count= self.net.get_normalized_expression(Dataset.adata)
        latent_space = self.net.get_latent_representation(Dataset.adata)
        latent_space = anndata.AnnData(X = latent_space, obs = Dataset.adata.obs)
        latent_space.obs[f"{self.class_key}_pred_scanvi"] = self.net.predict(Dataset.adata)
        return latent_space,corrected_count
    
    def save_net(self, save_path):
#         if not os.path.isdir(save_path):
#             os.mkdir(save_path) 
        self.net.save(save_path,overwrite=True)
                
     
class ScarchesScanvi_LCA:
    '''The following code is mainly adaptated from Lisa Sikkema notebook LCA_scArches_mapping_new_data_to_hlca.ipynb (found in /home/acollin/dca_permuted_workflow/LCA_scarches_reference/notebooks)'''
    def __init__(self, unlabeled_category = 'UNK', class_key='celltype',batch_key=None, n_samples_per_label =None, n_latent=50, gene_likelihood='zinb'):
        self.net = None
        self.vae = None
        self.query_data = sc.AnnData()
        self.batch_key = batch_key
        self.n_latent = n_latent
        self.gene_likelihood = gene_likelihood
        self.class_key = class_key
        self.unlabeled_category = unlabeled_category
        self.n_samples_per_label = n_samples_per_label
        self.scarches_info_path = "/home/acollin/dca_permuted_workflow/LCA_scarches_reference"
        self.path_gene_order = self.scarches_info_path + "/supporting_files/HLCA_scarches_gene_order.csv"
        self.path_embedding = self.scarches_info_path + "/data/HLCA_emb_and_metadata.h5ad"
        self.path_HLCA_celltype_info = self.scarches_info_path + "/supporting_files/HLCA_celltypes_ordered.csv"
        self.dir_ref_model = self.scarches_info_path + "/data/HLCA_reference_model"
        self.dir_testdata = self.scarches_info_path + "/test"
        self.reference_gene_order = pd.read_csv(self.path_gene_order)
        self.reference_embedding = sc.read_h5ad(self.path_embedding)
        self.batch_models = dict()
        self.query_batches = []
        self.surgery_epochs = 500
        self.early_stopping_kwargs_surgery = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "on": "full_dataset",
            "patience": 10,
            "threshold": 0.001,
            "reduce_lr_on_plateau": True,
            "lr_patience": 8,
            "lr_factor": 0.1,
        }
        self.batch_variable = []

    def make_net(self, Dataset):
        '''In this case, make_net unfortunately doesn't make the nets... '''
        adata_train = Dataset.adata
        self.query_data = data_import_and_cleaning.subset_and_pad_adata_object(adata_train, self.reference_gene_order)
        if (self.query_data.var.index == self.reference_gene_order.gene_symbol).all() or (
            self.query_data.var.index == self.reference_gene_order.gene_id).all():
            print("Gene order is correct.")
        else:
            print(
                "WARNING: your gene order does not match the order of the HLCA reference. Fix this before continuing!"
        )
        self.query_data.raw = self.query_data
        raw = self.query_data.raw.to_adata()
        raw.X = self.query_data.X
        self.query_data.raw = raw
        self.query_data.obs["scanvi_label"] = "unlabeled"
        if not self.batch_key:
            for b_k in ['manip', 'sample']:
                if b_k in adata_train.obs.columns:
                    self.batch_variable = b_k
        else :
            self.batch_variable = self.batch_key # the column name under which you stored your batch variable
        self.query_batches = sorted(self.query_data.obs[self.batch_variable].unique())
        print(self.query_batches)

        for batch in self.query_batches: # this loop is only necessary if you have multiple batches, but will also work for a single batch.
            print("Batch:", batch)
            query_subadata = self.query_data[self.query_data.obs[self.batch_variable] == batch,:].copy()
            print("Shape:", query_subadata.shape)
            # load model and set relevant variables:
            model = sca.models.SCANVI.load_query_data(
                query_subadata,
                self.dir_ref_model,
                freeze_dropout = True,
            )
            model._unlabeled_indices = np.arange(query_subadata.n_obs)
            model._labeled_indices = []
            self.batch_models = model


    def train_net(self, Dataset):
        for batch in self.query_batches:
            model = self.batch_models[batch]
            model.train(
                n_epochs_semisupervised=self.surgery_epochs,
                train_base_model=False,
                semisupervised_trainer_kwargs=dict(
                    metrics_to_monitor=["accuracy", "elbo"], 
                    weight_decay=0,
                    early_stopping_kwargs=self.early_stopping_kwargs_surgery
                ),
                frequency=1
            )
            self.batch_models[batch] = model

        
    def predict_net(self, Dataset):
        emb_df = pd.DataFrame(index=self.query_data.obs.index,columns=range(0,self.reference_embedding.shape[1]))
        for batch in self.query_batches: # from small to large datasets
            print(f"Working on {batch}...")
            query_subadata = self.query_data[self.query_data.obs[self.batch_variable] == batch,:].copy()
            model = self.batch_models[batch]
            query_subadata_latent = sc.AnnData(model.get_latent_representation(adata=query_subadata))
            # copy over .obs
            query_subadata_latent.obs = self.query_data.obs.loc[query_subadata.obs.index,:]
            emb_df.loc[query_subadata.obs.index,:] = query_subadata_latent.X
        corrected_count = sc.AnnData() # Placeholder for the corrected counts
        query_embedding = sc.AnnData(X=emb_df.values, obs=self.query_data.obs)

        query_embedding.obs['HLCA_or_query'] = "query"
        self.reference_embedding.obs['HLCA_or_query'] = "HLCA"
        combined_emb = self.reference_embedding.concatenate(query_embedding, index_unique=None) # index_unique="_", batch_key="dataset") # alternative

        cts_ordered = pd.read_csv(self.path_HLCA_celltype_info,index_col=0)
        # run k-neighbors transformer
        k_neighbors_transformer = scarches_label_transfer.weighted_knn_trainer(
            train_adata=self.reference_embedding,
            train_adata_emb="X", # location of our joint embedding
            n_neighbors=50,
            )    
        # perform label transfer
        labels, uncert = scarches_label_transfer.weighted_knn_transfer(
            query_adata=query_embedding,
            query_adata_emb="X", # location of our joint embedding
            label_keys="Level",
            knn_model=k_neighbors_transformer,
            ref_adata_obs = self.reference_embedding.obs.join(cts_ordered, on='ann_finest_level')
            )
        uncertainty_threshold = 0.2

        labels.rename(columns={f"Level_{lev}":f"Level_{lev}_transfered_label_unfiltered" for lev in range(1,6)},inplace=True)
        uncert.rename(columns={f"Level_{lev}":f"Level_{lev}_transfer_uncert" for lev in range(1,6)},inplace=True)
        
        combined_emb.obs = combined_emb.obs.join(labels)
        combined_emb.obs = combined_emb.obs.join(uncert)
        # convert to arrays instead of categoricals, and set "nan" to NaN 
        for col in combined_emb.obs.columns:
            if col.endswith("_transfer_uncert"):
                combined_emb.obs[col] = list(np.array(combined_emb.obs[col]))
            elif col.endswith("_transfered_label_unfiltered"):
                filtered_colname = col.replace("_unfiltered","")
                matching_uncert_col = col.replace("transfered_label_unfiltered","transfer_uncert")
                
                # also create a filtered version, setting cells with too high 
                # uncertainty levels to "Unknown"
                combined_emb.obs[filtered_colname] = combined_emb.obs[col]
                combined_emb.obs[filtered_colname].loc[combined_emb.obs[matching_uncert_col]>uncertainty_threshold] = "Unknown"
                # convert to categorical:
                combined_emb.obs[col] = pd.Categorical(combined_emb.obs[col])
                combined_emb.obs[filtered_colname] = pd.Categorical(combined_emb.obs[filtered_colname])
                # then replace "nan" with NaN (that makes colors better in umap)
                combined_emb.obs[col].replace("nan",np.nan,inplace=True)
                combined_emb.obs[filtered_colname].replace("nan",np.nan,inplace=True)

        return query_embedding,combined_emb
    
    def save_net(self, save_path):
#         if not os.path.isdir(save_path):
#             os.mkdir(save_path) 
        for batch, model in self.batch_models.items():
            surgery_path = os.path.join(save_path,batch)
            if not os.path.exists(surgery_path):
                os.makedirs(surgery_path)
            model.save(surgery_path, overwrite=True)
    
# class Autoencoder(Model):
#     def __init__(self, latent_dim):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim   
#         self.encoder = tf.keras.Sequential([
#           layers.Flatten(),
#           layers.Dense(latent_dim, activation='relu'),
#         ])
#         self.decoder = tf.keras.Sequential([
#           layers.Dense(784, activation='sigmoid'),
#           layers.Reshape((28, 28))
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#     return decoded

# autoencoder = Autoencoder(latent_dim)
