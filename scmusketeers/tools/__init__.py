# NB: `plot` is intentionally not imported here. It pulls in matplotlib/seaborn
# (the optional "workflow" dependency group); import scmusketeers.tools.plot
# explicitly where you need the plotting helpers.
from . import clust_compute, layers, loss, models, permutation, utils