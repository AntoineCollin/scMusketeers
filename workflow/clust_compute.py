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
