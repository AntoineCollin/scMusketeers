# sc_permut - Deep Learning single-cell 

Deep learning annotation of cell-types with permutation inforced autoencoder


## Summary

### Model structure


## Examples



## Installation

CheckAtlas can be downloaded from PyPI. However, the project is in an early development phase. We strongly recommend to use the developmental version.

### Install checkatlas development version

```bash
git clone git@github.com:becavin-lab/sc_permut.git
cd sc_permut
poetry install
```

It needs CUDA installed ...

### Install it from PyPI

```bash
pip install sc_permut
```


## Usage

The one liner way to run checkatlas is the following:

```bash
$ cd your_search_folder/
$ python -m checkatlas .
#or
$ checkatlas .
```

Or run it inside your python workflow.

```py
from checkatlas import checkatlas
checkatlas.run(path, atlas_list, multithread, n_cpus)
```