from typing import Iterable, Optional

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.metrics import *

# from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score,silhouette_samples,f1_score,matthews_corrcoef,cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def silhouette(
    adata, partition_key, obsm_representation: Optional[str] = None
):
    """
    By default, computes the average silhouette coefficient for the adata with respect
    to the clustering specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    The the average silhouette coefficient of the data with respect to the
    partition_key clustering
    """
    annotations = adata.obs[partition_key]
    if obsm_representation:
        count_repr = adata.obsm[obsm_representation]
        return silhouette_score(count_repr, annotations)
    else:
        original_count = adata.X
        return silhouette_score(original_count, annotations)


def silhouette_sample(
    adata, partition_key, obsm_representation: Optional[str] = None
):
    """
    By default, computes the average silhouette coefficient for the adata with respect
    to the clustering specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    An array containing every silhouette coefficient per sample
    """
    annotations = adata.obs[partition_key]
    if obsm_representation:
        count_repr = adata.obsm[obsm_representation]
        return silhouette_samples(count_repr, annotations)
    else:
        original_count = adata.X
        return silhouette_samples(original_count, annotations)


# specify a "subset" parameter corresponding to a type to subset the adata. The metric
# would be computed on this subset. However, this requires to be rather careful with
# the used annotations etc... Maybe it would be smarter to create another module for
# subset analysis or at least a get_subset which computes correct fields for subsets
# Just create a function which create a subset adata and would be used as an argument
# in the other function (use subset = 'subset_name')


def davies_bouldin(
    adata: AnnData, partition_key, obsm_representation: Optional[str] = None
):
    """
    By default, computes the Davies-Bouldin index for the adata with respect to the
    clustering specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    The Davies-Bouldin index of the data with respect to the partition_key clustering
    """
    annotations = adata.obs[partition_key]
    if obsm_representation:
        count_repr = adata.obsm[obsm_representation]
        return davies_bouldin_score(count_repr, annotations)
    else:
        original_count = adata.X
        return davies_bouldin_score(original_count, annotations)


def calinski_harabasz(
    adata: AnnData, partition_key, obsm_representation: Optional[str] = None
):
    """
    By default, computes the Davies-Bouldin index for the adata with respect to the
    clustering specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    The Davies-Bouldin index of the data with respect to the partition_key clustering
    """
    annotations = adata.obs[partition_key]
    if obsm_representation:
        count_repr = adata.obsm[obsm_representation]
        return calinski_harabasz_score(count_repr, annotations)
    else:
        original_count = adata.X
        return calinski_harabasz_score(original_count, annotations)


def dbcv(adata, partition_key, obsm_representation: Optional[str] = None):
    """
    By default, computes the DBCV index for the adata with respect to the clustering
    specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    The DBCV index of the data with respect to the partition_key clustering
    """
    annotations = adata.obs[partition_key]
    if obsm_representation:
        count_repr = adata.obsm[obsm_representation]
        return davies_bouldin_score(count_repr, annotations)
    else:
        original_count = adata.X
        return davies_bouldin_score(original_count, annotations)


def annotation_to_num(adata, reference, partition_key):
    """
    Transforms the annotations from categorical to numerical
    Parameters
    ----------
    adata
    partition_key
    reference
    Returns
    -------
    """
    annotation = adata.obs[partition_key].to_numpy()
    ref_annotation = adata.obs[reference].to_numpy()
    le = LabelEncoder()
    le.fit(ref_annotation)
    annotation = le.transform(annotation)
    ref_annotation = le.transform(ref_annotation)
    return ref_annotation, annotation


def balanced_accuracy(adata, reference, partition_key):
    if not all(
        [
            i in np.unique(adata.obs[reference])
            for i in np.unique(adata.obs[partition_key])
        ]
    ):
        print(
            f"Warning : Acc is being computing only on categories present in {reference}"
        )
        adata_sub = adata[
            adata.obs[partition_key].isin(np.unique(adata.obs[reference]))
        ]  #
        return balanced_accuracy_score(
            *annotation_to_num(adata_sub, reference, partition_key)
        )
    return balanced_accuracy_score(
        *annotation_to_num(adata, reference, partition_key)
    )


def accuracy(adata, reference, partition_key):
    if not all(
        [
            i in np.unique(adata.obs[reference])
            for i in np.unique(adata.obs[partition_key])
        ]
    ):
        print(
            f"Warning : Acc is being computing only on categories present in {reference}"
        )
        adata_sub = adata[
            adata.obs[partition_key].isin(np.unique(adata.obs[reference]))
        ]
        return accuracy_score(
            *annotation_to_num(adata_sub, reference, partition_key)
        )
    return accuracy_score(*annotation_to_num(adata, reference, partition_key))


def rand(adata, reference, partition_key, n_threshold=70000):
    """
    By default, computes the Rand index for the adata with respect to the clustering
    specified by partition_key compared to the reference clustering.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotation to be used.
    reference : str
        The key in adata.obs corresponding to the reference annotation to be used.
    n_threshold: int
        number of cells from which an approximation of ARI is computed. This has to be done because adjusted_rand_score
        raises a long scalar warning when there are too many cells
    Returns
    -------
    The Rand index of the data with respect to the partition_key clustering compared
    to the reference clustering.
    """
    if adata.n_obs > n_threshold:
        l = []
        for i in range(
            100
        ):  # We compute ARI on 100 different samples of the dataset and return the average
            ind = np.random.choice(adata.obs_names, n_threshold)
            l.append(
                adjusted_rand_score(
                    adata.obs[reference][ind], adata.obs[partition_key][ind]
                )
            )
        return np.mean(l)
    else:
        return adjusted_rand_score(
            adata.obs[reference], adata.obs[partition_key]
        )


def fowlkes_mallows(adata, reference, partition_key):
    """
    By default, computes the Fowlkes-Mallows score for the adata with respect to the
    clustering specified by partition_key compared to the reference clustering.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotation to be used.
    reference : str
        The key in adata.obs corresponding to the reference annotation to be used.
    Returns
    -------
    The Fowlkes-Mallows score of the data with respect to the partition_key clustering
    compared to the reference clustering.
    """
    return fowlkes_mallows_score(
        *annotation_to_num(adata, reference, partition_key)
    )


def nmi(adata, reference, partition_key):
    """
    By default, computes the Normalized Mutual Information for the adata with respect
    to the clustering specified by partition_key compared to the reference clustering.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotation to be used.
    reference : str
        The key in adata.obs corresponding to the reference annotation to be used.
    Returns
    -------
    The Normalized Mutual Information of the data with respect to the partition_key
    clustering compared to the reference clustering.
    """
    return adjusted_mutual_info_score(
        adata.obs[reference], adata.obs[partition_key]
    )


def vmeasure(adata, reference, partition_key):
    """
    By default, computes the V-Measure for the adata with respect to the clustering
    specified by partition_key compared to the reference clustering.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotation to be used.
    reference : str
        The key in adata.obs corresponding to the reference annotation to be used.
    Returns
    -------
    The V-Measure of the data with respect to the partition_key clustering compared
    to the reference clustering.
    """
    return v_measure_score(*annotation_to_num(adata, reference, partition_key))


def dunn(adata, partition_key: str, obsm_representation: Optional[str] = None):
    """
    By default, computes the Dunn index for the adata with respect to the clustering
    specified by partition_key.
    If given a value for obsm_representation, computes the index on the representation
    of the original data stored at adata.obsm.obsm_representation.
    Parameters
    ----------
    adata : anndata
        The corrected expression matrix
    partition_key : str
        The key in adata.obs corresponding to the annotations to be used.
    obsm_representation : str
        Key of adata.obsm containing a representation of the data for example 'umap' or
        'pca' etc...
    Returns
    -------
    The Dunn index of the data with respect to the partition_key clustering
    """


def make_weight(y_true):
    """
    return a weight for each element of y_true corresponding to the number of elements of its class.
    Balanced metrics should use 1/weights
    """
    y_true_s = pd.Series(y_true)
    ct_weights = y_true_s.value_counts()
    weights = np.array(y_true_s.replace(ct_weights))
    return weights


def to_1d(y):
    return np.asarray(y).reshape(
        -1,
    )


def balanced_f1_score(y_true, y_pred):
    y_true, y_pred = to_1d(y_true), to_1d(y_pred)
    weights = make_weight(y_true)
    return f1_score(y_true, y_pred, sample_weight=1 / weights, average="macro")


def balanced_matthews_corrcoef(y_true, y_pred):
    y_true, y_pred = to_1d(y_true), to_1d(y_pred)
    weights = make_weight(y_true)
    return matthews_corrcoef(y_true, y_pred, sample_weight=1 / weights)


def balanced_cohen_kappa_score(y_true, y_pred):
    y_true, y_pred = to_1d(y_true), to_1d(y_pred)
    weights = make_weight(y_true)
    return cohen_kappa_score(y_true, y_pred, sample_weight=1 / weights)


import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

"""
# Author: Xiong Lei
# Created Time : Thu 10 Jan 2019 07:38:10 PM CST
# File Name: metrics.py
# Description:
"""


def batch_entropy_mixing_score(
    data,
    batches,
    n_neighbors=100,
    n_pools=100,
    n_samples_per_pool=100,
    sub_population=None,
    verbose=True,
):
    """
    Calculate batch entropy mixing score

    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.

    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
    sub_population
        Indices in which to fetch the randomly chosen cells. Could correspond to a specific celltype for example. By default, None uses the whole dataset.

    Returns
    -------
    Batch entropy mixing score
    """

    #     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    if verbose:
        print("Computing neighborhood graph")
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    print(P)
    for t in range(n_pools):
        if verbose and t % 10 == 0:
            print(f"Entropy mixing ----- step {t}")
        if sub_population is None:
            indices = np.random.choice(
                np.arange(data.shape[0]), size=n_samples_per_pool
            )
        else:
            n_samples_per_pool = min(n_samples_per_pool, len(sub_population))
            indices = np.random.choice(sub_population, size=n_samples_per_pool)
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices].nonzero()[1][
                            kmatrix[indices].nonzero()[0] == i
                        ]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))


import scipy


def batch_entropy_mixing(
    adata, batch_key, use_rep=None, sub_key=None, sub_obs=None
):
    if use_rep:
        data = adata.obsm[use_rep]
    else:
        data = adata.X
    if type(data) == scipy.sparse.csr.csr_matrix:
        data = data.toarray()
    if sub_obs and sub_key:
        return batch_entropy_mixing_score(
            data,
            adata.obs[batch_key],
            sub_population=np.where(adata.obs[sub_key] == sub_obs)[0],
        )
    else:
        return batch_entropy_mixing_score(data, adata.obs[batch_key])


### From SCANVI Github
def nn_overlap(X1, X2, k=100):
    nne = NearestNeighbors(n_neighbors=k + 1, n_jobs=8)
    assert len(X1) == len(X2)
    n_samples = len(X1)
    nne.fit(X1)
    kmatrix_1 = nne.kneighbors_graph(X1) - scipy.sparse.identity(n_samples)
    nne.fit(X2)
    kmatrix_2 = nne.kneighbors_graph(X2) - scipy.sparse.identity(n_samples)

    # 1 - spearman correlation from knn graphs
    spearman_correlation = scipy.stats.spearmanr(
        kmatrix_1.A.flatten(), kmatrix_2.A.flatten()
    )[0]
    # 2 - fold enrichment
    set_1 = set(np.where(kmatrix_1.A.flatten() == 1)[0])
    set_2 = set(np.where(kmatrix_2.A.flatten() == 1)[0])
    fold_enrichment = (
        len(set_1.intersection(set_2))
        * n_samples**2
        / (float(len(set_1)) * len(set_2))
    )
    jaccard_index = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
    return spearman_correlation, fold_enrichment, jaccard_index


###


def knn_purity_single_dataset(adata):
    """
    Compares two embeddings of the same data in terms of knn similarity
    """


def knn_purity(reference, embedding, embedding_key):
    """
    Compares the purity of an embedding compared to a reference
    """


###
# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


def lisi_avg(X, labels):
    print(labels)
    metadata = pd.DataFrame({"labels": labels})
    return np.mean(compute_lisi(X, metadata, ["labels"]))


def lisi_df(X, labels):
    metadata = pd.DataFrame({"labels": labels})
    return compute_lisi(X, metadata, ["labels"])


def compute_lisi(
    X: np.array,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30,
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(
        n_neighbors=perplexity * 3, algorithm="kd_tree"
    ).fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(
            distances.T, indices.T, labels, n_categories, perplexity
        )
        lisi_df[:, i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson
