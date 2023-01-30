import anndata
from anndata import AnnData
import numpy as np
from sklearn.metrics import davies_bouldin_score, adjusted_mutual_info_score, \
    adjusted_rand_score, fowlkes_mallows_score, v_measure_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, plot_confusion_matrix,silhouette_samples

from sklearn.preprocessing import LabelEncoder


def silhouette(adata, partition_key, obsm_representation: str = None):
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
    
def silhouette_sample(adata, partition_key, obsm_representation: str = None):
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


def davies_bouldin(adata : AnnData, partition_key, obsm_representation: str = None):
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


def calinski_harabasz(adata : AnnData, partition_key, obsm_representation: str = None):
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


def dbcv(adata, partition_key, obsm_representation: str = None):
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
    if not all([i in np.unique(adata.obs[reference]) for i in np.unique(adata.obs[partition_key])]):
        print(f'Warning : Acc is being computing only on categories present in {reference}')
        adata_sub = adata[adata.obs[partition_key].isin(np.unique(adata.obs[reference]))] # 
        return balanced_accuracy_score(*annotation_to_num(adata_sub, reference, partition_key))
    return balanced_accuracy_score(*annotation_to_num(adata, reference, partition_key))


def accuracy(adata, reference, partition_key):
    if not all([i in np.unique(adata.obs[reference]) for i in np.unique(adata.obs[partition_key])]):
        print(f'Warning : Acc is being computing only on categories present in {reference}')
        adata_sub = adata[adata.obs[partition_key].isin(np.unique(adata.obs[reference]))] 
        return accuracy_score(*annotation_to_num(adata_sub, reference, partition_key))
    return accuracy_score(*annotation_to_num(adata, reference, partition_key))


def rand(adata, reference, partition_key):
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
    Returns
    -------
    The Rand index of the data with respect to the partition_key clustering compared
    to the reference clustering.
    """
    return adjusted_rand_score(*annotation_to_num(adata, reference, partition_key))


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
    return fowlkes_mallows_score(*annotation_to_num(adata, reference, partition_key))


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
    return adjusted_mutual_info_score(*annotation_to_num(adata, reference, partition_key))


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



def dunn(adata , partition_key: str, obsm_representation: str = None):
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

import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

"""
# Author: Xiong Lei
# Created Time : Thu 10 Jan 2019 07:38:10 PM CST
# File Name: metrics.py
# Description:
"""

def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
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
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    print(batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    print(P)
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))

def batch_entropy_mixing(adata, batch_key):
    return batch_entropy_mixing_score(adata.X, adata.obs[batch_key])