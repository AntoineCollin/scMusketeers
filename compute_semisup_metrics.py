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

rfh = run_file_handler(working_dir = working_dir)
id_list = rfh.query_yaml(use_TEST = True,semi_sup = True)
aw = AnalysisWorkflow(working_dir = working_dir, id_list=[ID for ID in id_list if (ID not in [1200,2230,2231,2232]) and (ID >500) ])
aw.load_latent_spaces()

aw.compute_metrics()