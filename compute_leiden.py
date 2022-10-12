import nbformat as nbf
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import numpy as np
import os
from workflow.clust_compute import *
import scanpy as sc
from workflow.runfile_handler import run_file_handler
from workflow.analysis import AnalysisWorkflow
from tqdm.notebook import trange, tqdm
import seaborn as sns
import matplotlib.pyplot as plt



        


working_dir = '/home/acollin/dca_permuted_workflow'
rfh = run_file_handler(working_dir = working_dir)

### Fake only
ID_to_plot = pd.read_csv('/home/acollin/dca_permuted_workflow/metrics/best_htap_ID_sigmoid.csv', index_col = 0)
ID_to_plot = ID_to_plot.to_numpy().flatten()

aw_htap = AnalysisWorkflow(working_dir = working_dir, id_list=ID_to_plot)
aw_htap.load_latent_spaces()

to_remove = []
for wf_id, latent in aw_htap.latent_spaces.items():
    if latent.n_obs == 0:
        to_remove.append(wf_id)

ID_to_plot = list(set(ID_to_plot) - set(to_remove))
aw_htap = AnalysisWorkflow(working_dir = working_dir, id_list=ID_to_plot)
aw_htap.load_latent_spaces()

for wf_id, latent_space in aw_htap.latent_spaces.items():
    sc.tl.leiden(latent_space,neighbors_key = 'neighbors_dca_permuted')
    latent_space.write(aw_htap.workflow_list[wf_id].adata_path)
