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

## Fake and True

htap_wf_ID = rfh.query_yaml(dataset_name = 'htap',ae_type = 'normal')

aw_htap = AnalysisWorkflow(working_dir = working_dir, id_list=htap_wf_ID)
aw_htap.load_latent_spaces()

to_remove = []
for wf_id, latent in aw_htap.latent_spaces.items():
    if len(latent.obs['celltype_pred'].unique()) <2:
        to_remove.append(wf_id)
            
ID = rfh.query_yaml(dataset_name = 'htap', ae_type ='normal')
ID = list(set(ID) - set(to_remove))

aw_htap = AnalysisWorkflow(working_dir = working_dir, id_list=ID)
aw_htap.load_latent_spaces()

# aw_htap.subsample_true_false()

aw_htap.compute_metrics(verbose = True)

aw_htap.metrics_table.to_csv(working_dir + '/metrics/htap_metrics_normalAE.csv')
