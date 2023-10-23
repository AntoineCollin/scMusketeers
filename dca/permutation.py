import scanpy as sc
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import sklearn
import tensorflow as tf
import time

import random

def make_random_seed():
    random_seed = time.time()
    random_seed = int((random_seed*10 - int(random_seed*10)) * 10e7)
    return random_seed

def random_derangement(array):
    while True:
        array = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if array[p] == j:
                break
            else:
                array[j], array[p] = array[p], array[j]
        else:
            if array[0] != 0:
                return array

def make_training_pairs(ind_classes, n_perm):
    """
    Creates n_perm permutations of indices for the indices of a particular class.
    ind_classes : the dataframe index of interest for a particular class
    n_perm : number of permutations
    """
    n_c = len(ind_classes)
    # X_1 = [[ind_classes[i]] * n_perm for i in range(n_c)] # Duplicate the index
    X_1 = ind_classes * n_perm
    # X_1 = [x for sublist in X_1 for x in sublist] 
    X_perm = [sklearn.utils.shuffle(ind_classes, random_state = make_random_seed()) for i in range(n_perm)] # The corresponding permuted value
    X_perm = [x for sublist in X_perm for x in sublist]
    return X_1, X_perm

def make_training_set(y, n_perm,same_class_pct=None,unlabeled_category='UNK'):
    """
    Creates the total list of permutations to be used by the generator to create shuffled training batches
    same_class_pct : When using contrastive loss, indicates the pct of samples to permute within their class. set to None when not using contrastive loss
    """
    permutations = [[],[]]
    y = np.array(y).astype(str)
    print('switching perm')
    classes = np.unique(y)
    ind_c = list(np.where(y)[0])
    if same_class_pct :
        ind_c_same, ind_c_diff = train_test_split(ind_c, train_size = same_class_pct, random_state=make_random_seed()) # shuffling same_class_pct % in same class and the rest at random
        X1, Xperm = make_training_pairs(ind_c_diff, n_perm)
        permutations[0] += X1
        permutations[1] += Xperm
    else :
         ind_c_same = ind_c
    for classe in classes:
        if classe == unlabeled_category: # We mark unknown classes with the 'UNK' tag
            ind_c = list(set(list(np.where(y == classe)[0])) & set(ind_c_same))
            X1 = make_training_pairs(ind_c, n_perm)[0]
            permutations[0] += X1
            permutations[1] += X1 # if the class is unknown, we reconstruct the same cell
        else :
            ind_c = list(set(list(np.where(y == classe)[0])) & set(ind_c_same))
            X1, Xperm = make_training_pairs(ind_c, n_perm)
            permutations[0] += X1
            permutations[1] += Xperm
    return [(a,b) for a,b in zip(permutations[0], permutations[1])]


def batch_generator_training_permuted(adata, class_key, batch_size, n_perm, use_raw_as_output,change_perm = True,same_class_pct=None, batch_key=None, unlabeled_category='UNK'):
    """
    Permuted batch generation for dca. adata should have an obs field containing size factors.adata should have been processed by the dca normalize function therefore it should already be in dense form.
    """
    print(adata)
    X = adata.X
    if use_raw_as_output:
        X_out = adata.raw.X
    else :
        X_out = adata.X
    sf = adata.obs['size_factors']
    y = adata.obs[class_key]
    perm_indices = make_training_set(y = y, n_perm = n_perm,same_class_pct=same_class_pct, unlabeled_category = unlabeled_category)
    samples_per_epoch = len(perm_indices)
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    perm_indices = sklearn.utils.shuffle(perm_indices, random_state = make_random_seed()) # Change to sklearn.utils.shuffle
    ind_in = [ind[0] for ind in perm_indices]
    ind_out = [ind[1] for ind in perm_indices]
    if same_class_pct: # In this case, we're using contrastive AE so we need to yield a similarity indicator
        sim = y[ind_in].values == y[ind_out].values
    #X =  X[shuffle_index, :]
    #y =  y[shuffle_index]
    if batch_key: # In this case, we're using batch removal AE so we need to yield the batch ID
        ohe = OneHotEncoder()
        batch_ID = adata.obs[batch_key] # batch_ID refers to the batches of the single cell experiment whereas batch_size refers to the training batches of the training procedure
        batch_ID = ohe.fit_transform(np.array(batch_ID).reshape(-1, 1)).toarray()
    i=0
    while 1:
        if counter == samples_per_epoch//batch_size :
            index_in_batch = ind_in[batch_size*counter:]
            ind_out_batch = ind_out[batch_size*counter:]
        else :
            index_in_batch = ind_in[batch_size*counter:batch_size*(counter+1)]
            ind_out_batch = ind_out[batch_size*counter:batch_size*(counter+1)]
        # X_in_batch = X[ind_in,:].todense()
        # X_in_batch = np.array(X_in_batch).reshape(X_in_batch.shape[0], X_in_batch.shape[1], 1)        
        # X_out_batch = X[X_out_batch,:].todense()
        # X_out_batch = np.array(X_out_batch).reshape(X_out_batch.shape[0], X_out_batch.shape[1], 1)
        sf_in_batch = sf[index_in_batch]
        X_in_batch = X[index_in_batch,:]
        X_out_batch = X_out[ind_out_batch,:]
        counter += 1
        if same_class_pct: # We're using a contrastive loss so we need to yield similarity
            sim_in_batch = sim[index_in_batch]
            yield({'count': X_in_batch,'size_factors' : sf_in_batch, 'similarity' : sim_in_batch}, X_out_batch) #first dim is the number of batches, next dims are the shape of input
        if batch_key:
            batch_ID_in_batch = batch_ID[index_in_batch,:] # The batch ID corresponding to the input cells
            batch_ID_in_batch=1-batch_ID_in_batch
            yield({'count': X_in_batch,'size_factors' : sf_in_batch}, {'reconstruction': X_out_batch,'batch_removal':batch_ID_in_batch})
        else:
            yield({'count': X_in_batch,'size_factors' : sf_in_batch}, X_out_batch) #first dim is the number of batches, next dims are the shape of input
        if (counter == samples_per_epoch//batch_size and samples_per_epoch % batch_size == 0) or (counter > number_of_batches):
            if change_perm:
                perm_indices = make_training_set(y = y, n_perm = n_perm,same_class_pct=same_class_pct, unlabeled_category = unlabeled_category)
            perm_indices = sklearn.utils.shuffle(perm_indices, random_state = make_random_seed()) # Change to sklearn.utils.shuffle
            ind_in = [ind[0] for ind in perm_indices]
            ind_out = [ind[1] for ind in perm_indices]
            if same_class_pct:
                sim = y[ind_in].values == y[ind_out].values
            debug = pd.DataFrame({'in':ind_in,'out':ind_out})
            print(make_random_seed())
            # debug.to_csv(f'/home/acollin/jobs/dca_jobs/workflow_jobs/log_debug/log{i}.csv')
            i+=1
            counter=0
            
