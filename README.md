# sc_permut

Deep learning annotation of cell-types with permutation inforced autoencoder



## Summary

### Parse Scanpy, Seurat and CellRanger objects

Fast crawl through your folder and detect Seurat (.rds), Scanpy (.h5ad) or cellranger (.h5) atlas files.

## Usage

The one liner way to run checkatlas is the following: 

```bash
$ cd your_search_folder/
$ python -m sc_permut .
#or
$ checkatlas .
```

Or run it inside your python workflow.

```py
from sc_permut import sc_permut
sc_permut.run(path, atlas_list, multithread, n_cpus)
```


## Development

Read the [CONTRIBUTING.md](docs/contributing.md) file.