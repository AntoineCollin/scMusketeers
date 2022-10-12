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

## All

htap_wf_ID = rfh.query_yaml(dataset_name = 'htap', predictor_activation= 'sigmoid')

aw_htap = AnalysisWorkflow(working_dir = working_dir, id_list=htap_wf_ID)
aw_htap.load_latent_spaces()

aw_htap.compute_metrics(verbose = True)

aw_htap.metrics_table.to_csv(working_dir + '/metrics/htap_all_sigmoid_metrics.csv')

