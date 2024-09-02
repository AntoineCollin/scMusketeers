# scMusketeers
Deep learning annotation of cell-types with permutation inforced autoencoder



## Summary

Single cell gene expression atlases are now central to explore the cellular diversity arising at the scale of organisms or organs. The emergence of ever larger datasets are benefiting from the rapid development of deep learning technologies in the field. The constitution of large datasets raises several big challenges due to the presence of highly imbalanced cell types or the existence of large batch effects, which need to be adressed in order to annotate properly newer data derived from very small subsets, transfer a model from one dataset to another.

We developed scPermut to learn an optimal dimension-reduced representation, while preserving the  information essential to meeting the above-mentioned challenges. The architecture of scPermut is made of three modules. The first module is an autoencoder which provides a reduced representation, while removing noise, and which allows a better data reconstruction. A classifier module with its focal loss can be combined to predict more accurately small cell types. This second module also supports transferring the learnt model to other datasets. The third module is an adversarial domain adaptation (DANN) module that corrects batch effect.

We extensively optimized scPermut hyperparameters, by conducting a precise ablation study to assess model's performance. We show that our model is at least on par with State-Of-The-Art models, and even outperforms them on most challenges. This was more thoroughly documented by comparing the different approaches in 12 datasets that differ in size, number of cell types, number or distinct experimental modes.

We anticipate that the generic modular framework that we provide can be easily adaptable to other fields of large-scale biology.



## Install

You can install sc_permut with Pypi:

```bash
$ pip install scpermut
```
with conda

```bash
$ conda -c bioconda scpermut
```

with docker


## Examples

sc_permut can be used for different task in integration and annotation of single-cell atlas. 

Here are 4 different examples:

- Label transfer between batch

```bash
$ scpermut label my_atlas --class_key celltype --batch_key donor
```

- Label transfer for completing atlas annotation


```bash
$ conda -c bioconda scpermut
```

with docker


TO DO : Add example atlas in the github or Zenodo


Read the [CONTRIBUTING.md](docs/contributing.md) file.
