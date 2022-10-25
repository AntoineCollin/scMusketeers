import nbformat as nbf
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import metrics
from workflow.runfile_handler import run_file_handler
from workflow.analysis import AnalysisWorkflow
from tqdm.notebook import trange, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

try :
    from .load import load_runfile
    from .dataset import Dataset
    from .predictor import MLP_Predictor
    from .model import DCA_Permuted
    from .workflow import Workflow
    from .runfile_handler import RunFile
    from .clust_compute import *
except ImportError:
    from workflow.load import load_runfile
    from workflow.dataset import Dataset
    from workflow.predictor import MLP_Predictor
    from workflow.model import DCA_Permuted
    from workflow.workflow import Workflow
    from workflow.runfile_handler import RunFile
    from workflow.clust_compute import *

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

working_dir = '/home/acollin/dca_permuted_workflow'

metrics_list = ["balanced_accuracy_scores",
"balanced_accuracy_scores_test",
"balanced_accuracy_scores_val",
"balanced_accuracy_scores_train",
"accuracy_scores",
"accuracy_scores_test",
"accuracy_scores_val",
"accuracy_scores_train",
#"silhouette_true",
#"silhouette_pred",
"davies_bouldin_true",
"davies_bouldin_pred",
"nmi"]

runfile_df = pd.read_csv('/home/acollin/dca_permuted_workflow/runfile_dir/runfile_list.csv', index_col = 0)
id_list = list(runfile_df[runfile_df['workflow_ID']>=971]['workflow_ID'])
aw = AnalysisWorkflow(working_dir = working_dir, id_list=[ID for ID in id_list if ID not in [1200,2230,2231,2232]])
aw.load_latent_spaces()

for ID, wf in aw.workflow_list.items():
    metric_series = pd.read_csv(wf.metric_path, index_col = 0)
    metric_clone = metric_series.copy()
    print(f'computing ID {ID}')
    for metric in metrics_list:
        if (metric not in metric_series.columns) or (metric_series.isna().loc[ID,metric]):
            print('computing metric')
            if metric == 'balanced_accuracy_scores':
                try :
                    metric_series.loc[ID,metric] = balanced_accuracy(adata=wf.latent_space, 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except :
                    print('mistakes were made')
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'balanced_accuracy_scores_test':
                try :
                    metric_series.loc[ID,metric] = balanced_accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'test'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'balanced_accuracy_scores_val':
                try :
                    metric_series.loc[ID,metric] = balanced_accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'val'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'balanced_accuracy_scores_train':
                try :
                    metric_series.loc[ID,metric] = balanced_accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'train'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan
            
            elif metric == 'accuracy_scores':
                try :
                    metric_series.loc[ID,metric] = accuracy(adata=wf.latent_space, 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'accuracy_scores_test':
                try :
                    metric_series.loc[ID,metric] = accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'test'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'accuracy_scores_val':
                try :
                    metric_series.loc[ID,metric] = accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'val'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'accuracy_scores_train':
                try :
                    metric_series.loc[ID,metric] = accuracy(adata=wf.latent_space[wf.latent_space.obs['train_split'] == 'train'], 
                                                                partition_key=f'{wf.class_key}_pred',
                                                                reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'silhouette_true':
                try :
                    metric_series.loc[ID,metric] = silhouette(adata=wf.latent_space, partition_key=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'silhouette_pred':
                try :
                    metric_series.loc[ID,metric] = silhouette(adata=wf.latent_space, partition_key=f'{wf.class_key}_pred')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'silhouette_true':
                try :
                    metric_series.loc[ID,metric] = davies_bouldin(adata=wf.latent_space, partition_key=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'silhouette_pred':
                try :
                    metric_series.loc[ID,metric] = davies_bouldin(adata=wf.latent_space, partition_key=f'{wf.class_key}_pred')
                except : 
                    metric_series.loc[ID,metric] = np.nan

            elif metric == 'nmi':
                try :
                    metric_series.loc[ID,metric] = nmi(adata=wf.latent_space, partition_key=f'{wf.class_key}_pred',reference=f'true_{wf.class_key}')
                except : 
                    metric_series.loc[ID,metric] = np.nan
                    

    if not metric_series.equals(metric_clone):
        metric_series.to_csv(wf.metric_path)
        print(f'metric_series saved for ID {ID}')
        aw.metric_results_df.update(metric_series)
        aw.metric_results_df.to_csv(aw.metric_results_path)

