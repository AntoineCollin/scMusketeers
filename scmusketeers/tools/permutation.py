import gc
import os
import random
import subprocess as sp
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


def get_gpu_memory(txt):
    # command = "nvidia-smi --query-gpu=memory.used --format=csv"
    # memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    memory_free_values = tf.config.experimental.get_memory_info("GPU:0")[
        "current"
    ]
    print(f"GPU memory usage in {txt} HERE: {memory_free_values/1e6} MB ")


import random


def make_random_seed():
    random_seed = time.time()
    random_seed = int((random_seed * 10 - int(random_seed * 10)) * 10e7)
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
    X_perm = [
        sklearn.utils.shuffle(ind_classes, random_state=make_random_seed())
        for i in range(n_perm)
    ]  # The corresponding permuted value
    X_perm = [x for sublist in X_perm for x in sublist]
    return X_1, X_perm


def make_training_set(
    y, n_perm, same_class_pct=None, unlabeled_category="UNK"
):
    """
    Creates the total list of permutations to be used by the generator to create shuffled training batches
    same_class_pct : When using contrastive loss, indicates the pct of samples to permute within their class. set to None when not using contrastive loss
    """
    permutations = [[], []]
    print("switching perm")

    # y = np.array(y).astype(str) If passing labels as string

    y_cl = np.asarray(y.argmax(axis=1)).flatten()  # convert one_hot to classes
    classes = np.unique(y_cl)
    ind_c = list(np.where(y_cl)[0])
    if same_class_pct:
        ind_c_same, ind_c_diff = train_test_split(
            ind_c, train_size=same_class_pct, random_state=make_random_seed()
        )  # shuffling same_class_pct % in same class and the rest at random
        X1, Xperm = make_training_pairs(ind_c_diff, n_perm)
        permutations[0] += X1
        permutations[1] += Xperm
    else:
        ind_c_same = ind_c
    for classe in classes:
        if (
            classe == unlabeled_category
        ):  # We mark unknown classes with the 'UNK' tag
            ind_c = list(
                set(list(np.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1 = make_training_pairs(ind_c, n_perm)[0]
            permutations[0] += X1
            permutations[
                1
            ] += X1  # if the class is unknown, we reconstruct the same cell
        else:
            ind_c = list(
                set(list(np.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1, Xperm = make_training_pairs(ind_c, n_perm)
            permutations[0] += X1
            permutations[1] += Xperm
    return [(a, b) for a, b in zip(permutations[0], permutations[1])]


def make_training_set_tf(
    y, n_perm, same_class_pct=None, unlabeled_category="UNK"
):
    """
    Creates the total list of permutations to be used by the generator to create shuffled training batches
    same_class_pct : When using contrastive loss, indicates the pct of samples to permute within their class. set to None when not using contrastive loss
    """
    permutations = [[], []]
    print("switching perm")
    # y = np.array(y).astype(str) If passing labels as string
    y_cl = tf.math.argmax(y, axis=1)
    classes = tf.unique(y_cl)
    ind_c = tf.where(y_cl)

    if same_class_pct:
        ind_c_same, ind_c_diff = train_test_split(
            ind_c, train_size=same_class_pct, random_state=make_random_seed()
        )  # shuffling same_class_pct % in same class and the rest at random
        X1, Xperm = make_training_pairs(ind_c_diff, n_perm)
        permutations[0] += X1
        permutations[1] += Xperm
    else:
        ind_c_same = ind_c
    for classe in classes:
        if (
            classe == -1
        ):  # unlabeled_category # We mark unknown classes with the 'UNK' tag
            ind_c = list(
                set(list(tf.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1 = make_training_pairs(ind_c, n_perm)[0]
            permutations[0] += X1
            permutations[
                1
            ] += X1  # if the class is unknown, we reconstruct the same cell
        else:
            ind_c = list(
                set(list(tf.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1, Xperm = make_training_pairs(ind_c, n_perm)
            permutations[0] += X1
            permutations[1] += Xperm
    return [(a, b) for a, b in zip(permutations[0], permutations[1])]


def batch_generator_training_permuted(
    X,
    y,
    sf,
    batch_ID=None,
    batch_size=128,
    ret_input_only=False,
    n_perm=1,  # TODO : remove n_perm. We always use n_perm=1, one epoch = one pass of the dataset
    change_perm=True,
    same_class_pct=None,
    unlabeled_category="UNK",
    use_perm=True,
):
    """
    Permuted batch generation for dca. adata should have an obs field containing size factors.adata should have been
    processed by the dca normalize function therefore it should already be in dense form.

    X : The count matrix object to generate, used only for size_factors
    y : The one hot classes corresponding to the count matrix
    sf : The size factors corresponding to the count matrix
    batch_ID : The batches corresponding to the count matrix
    batch_size : the size of the batches to yield
    use_raw_as_output : wether to yield the raw data as output, independantly form the data yielded as input
    ret_input_only : weither to return only inputs. Used in the case of custom training.
    change_perm : changes the permutation used once the whole dataset has been processed once. Always use true
    same_class_pct : defines the percentage of permutation which are made within class. Defaults is all permutation are within class.
    Otherwise, some observations will be matched with an observation from a different class. Mostly used for the contrastive loss
    batch_key : the key of the batches. Used to yield the batch as well when using DANN
    unlabeled_category : unknwon classes in adata.obs['class_key']
    use_perm : weither to use permutation or not. If False, this is simply an AE generator which yields X for input and output batch by batch
    """
    # sf = np.array(sf)
    X_out = X  # .copy()
    # gc.collect()
    perm_indices = make_training_set(
        y=y,
        n_perm=n_perm,
        same_class_pct=same_class_pct,
        unlabeled_category=unlabeled_category,
    )
    # perm_indices = make_training_set_tf(y = y, n_perm = n_perm,same_class_pct=same_class_pct, unlabeled_category = unlabeled_category)

    samples_per_epoch = len(perm_indices)
    number_of_batches = samples_per_epoch / batch_size
    counter = 0
    perm_indices = sklearn.utils.shuffle(
        perm_indices, random_state=make_random_seed()
    )  # Change to sklearn.utils.shuffle
    ind_in = [ind[0] for ind in perm_indices]
    if use_perm:
        ind_out = [ind[1] for ind in perm_indices]
    else:
        ind_out = ind_in  # Here, we're not using permutations
    if (
        same_class_pct
    ):  # In this case, we're using contrastive AE so we need to yield a similarity indicator
        sim = y[ind_in].values == y[ind_out].values
    # X =  X[shuffle_index, :]
    # y =  y[shuffle_index]
    i = 0
    while 1:
        if counter == samples_per_epoch // batch_size:
            index_in_batch = ind_in[batch_size * counter :]
            ind_out_batch = ind_out[batch_size * counter :]
        else:
            index_in_batch = ind_in[
                batch_size * counter : batch_size * (counter + 1)
            ]
            ind_out_batch = ind_out[
                batch_size * counter : batch_size * (counter + 1)
            ]
        # X_in_batch = X[ind_in,:].todense()
        # X_in_batch = np.array(X_in_batch).reshape(X_in_batch.shape[0], X_in_batch.shape[1], 1)
        # X_out_batch = X[X_out_batch,:].todense()
        # X_out_batch = np.array(X_out_batch).reshape(X_out_batch.shape[0], X_out_batch.shape[1], 1)
        sf_in_batch = sf.iloc[index_in_batch]
        y_in_batch = y[index_in_batch]
        X_in_batch = X[index_in_batch, :]
        X_out_batch = X_out[ind_out_batch, :]
        if (type(X) == scipy.sparse.csr_matrix) or (
            type(X) == scipy.sparse.csc_matrix
        ):
            X_in_batch = X_in_batch.todense()
            X_out_batch = X_out_batch.todense()
        counter += 1
        if (
            same_class_pct
        ):  # We're using a contrastive loss so we need to yield similarity
            sim_in_batch = sim[index_in_batch]
            if ret_input_only:
                yield (
                    {
                        "counts": X_in_batch,
                        "size_factors": sf_in_batch,
                        "similarity": sim_in_batch,
                    }
                )
            else:
                yield (
                    {
                        "counts": X_in_batch,
                        "size_factors": sf_in_batch,
                        "similarity": sim_in_batch,
                    },
                    {X_out_batch},
                )  # first dim is the number of batches, next dims are the shape of input
        if batch_ID is not None:
            batch_ID_in_batch = batch_ID[
                index_in_batch, :
            ]  # The batch ID corresponding to the input cells
            # batch_ID_in_batch = 1-batch_ID_in_batch # ???
            if ret_input_only:
                yield ({"counts": X_in_batch, "size_factors": sf_in_batch})
            else:
                # yield({'counts': X_in_batch,'size_factors' : sf_in_batch}, {'batch_discriminator':batch_ID_in_batch,
                #                                                            'reconstruction': X_out_batch})
                yield (
                    {"counts": X_in_batch, "size_factors": sf_in_batch},
                    {
                        "classifier": y_in_batch,
                        "batch_discriminator": batch_ID_in_batch,
                        "reconstruction": X_out_batch,
                    },
                )
        else:
            if ret_input_only:
                yield ({"counts": X_in_batch, "size_factors": sf_in_batch})
            else:
                yield (
                    {"counts": X_in_batch, "size_factors": sf_in_batch},
                    X_out_batch,
                )  # first dim is the number of batches, next dims are the shape of input
        if (
            counter == samples_per_epoch // batch_size
            and samples_per_epoch % batch_size == 0
        ) or (
            counter > number_of_batches
        ):  # The entire dataset has been processed, we reset the state of the generator
            if change_perm:
                perm_indices = make_training_set(
                    y=y,
                    n_perm=n_perm,
                    same_class_pct=same_class_pct,
                    unlabeled_category=unlabeled_category,
                )
                gc.collect()
            random_seed = make_random_seed()
            perm_indices = sklearn.utils.shuffle(
                perm_indices, random_state=random_seed
            )  # Shuffles the orders of observations
            ind_in = [ind[0] for ind in perm_indices]
            if use_perm:
                ind_out = [ind[1] for ind in perm_indices]
            else:
                ind_out = ind_in  # Here, we're not using permutations
            if same_class_pct:
                sim = y[ind_in].values == y[ind_out].values
            debug = pd.DataFrame({"in": ind_in, "out": ind_out})
            print(random_seed)
            # debug.to_csv(f'/home/acollin/jobs/dca_jobs/workflow_jobs/log_debug/log{i}.csv')
            i += 1
            counter = 0
