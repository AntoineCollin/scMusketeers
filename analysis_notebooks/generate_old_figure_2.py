import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import sys
import argparse

parser = argparse.ArgumentParser(description="Generate old Figure 2 plots")
parser.add_argument("--stats", action="store_true", help="Include statistical annotations")
args = parser.parse_args()

# Setup working directory dynamically to match the exact environment
working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/'
fig_dir = os.path.join(working_dir, 'analysis_notebooks', 'figures_review_old') + '/'

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Import the aesthetic dictionaries exactly as used
sys.path.append(os.path.join(working_dir, 'analysis_notebooks'))
from figure_scripts.notebook_tools import test_fold_selection, aestetic_data_name

print("Loading legacy pickle table...")
pkl_path = os.path.join(working_dir, 'analysis_notebooks', 'figures_review_old', 'task_1.pkl')
if not os.path.exists(pkl_path):
    print(f"Error: Pickle file not found at {pkl_path}")
    sys.exit(1)

runs_table_df = pd.read_pickle(pkl_path)
print(f"Loaded {len(runs_table_df)} rows from legacy pickle.")

# Figure 2 groupings
diff_dict = {'homo': ['tosti_2021',  'yoshida_2021', 'htap', 'ajrccm_by_batch'], 
           'assays': ['dominguez_2022_lymph', 'dominguez_2022_spleen', 'tabula_2022_spleen'],
           'suspension': ['koenig_2022', 'litvinukova_2020', 'lake_2021'],
           'datasets': ['hlca_par_dataset_harmonized','hlca_trac_dataset_harmonized']}

# Normalizing scPermut_default exactly as in the notebook
runs_table_df.loc[(runs_table_df['model'] == 'scPermut') & (runs_table_df['use_hvg'] == 3000),'model'] = 'scPermut_default'

t1 = pd.DataFrame()
task_dict = {}

print("Filtering and mapping dataset splits...")
for diff in diff_dict:
    if diff != 'dataset':  # Strict replica of line 866 condition
        sub_task = pd.DataFrame()
        for dataset_name in diff_dict[diff]:
            fold = test_fold_selection.get(dataset_name, 0)
            
            # Legacy robust query
            task_1 = runs_table_df.query("task == 'task_1'")\
                                  .query(f"dataset_name == '{dataset_name}'")\
                                  .query(f"test_fold_nb == {fold}")\
                                  .query("deprecated_status != True")
            
            task_1 = task_1.loc[task_1['model'] != 'scPermut']
            task_1 = task_1.loc[~((task_1['model'] == 'scPermut_default') & (task_1['training_scheme'] != 'training_scheme_8')),:] 
            task_1 = task_1.loc[~((task_1['model'] == 'scPermut_default') & (task_1['debug_status'] != 'fixed_1')),:]
            
            t1 = pd.concat([t1, task_1])
            sub_task = pd.concat([sub_task, task_1])
        task_dict[diff] = sub_task.copy()

aestetic_model_name = {'scPermut_default' : '1.scMusketeers',
                       'scanvi' : '2.scANVI',
                       'uce' : '3.UCE', 
                       'harmony_svm' : '4.Harmony',
                       'pca_svm' : '5.PCA',
                       'celltypist' : '6.Celltypist',
                       'scmap_cells' : '7.scmap - cells',
                       'scmap_cluster' : '8.scmap - cluster'}

colors = {
          'scPermut_default' : "#B15240",
          'scanvi':"#C78C3B" ,
          'uce':"#D3A53C" ,
          'harmony_svm': "#5B9DC7",
          'pca_svm':"#264D74",
          'celltypist': "#75BAD3",
          'scmap_cells':"#607F6A" ,
          'scmap_cluster':"#707C45"}

colors = {aestetic_model_name[k]: v for k, v in colors.items() if k in aestetic_model_name}

metric_names = {'balanced_acc': 'Balanced Accuracy'}
split = 'test'

xsizes = [24, 15, 15, 10]
ysizes = [5, 4, 4, 3]

print("Generating old Figure 2 plots...")
for met in ['balanced_acc']:
    i = 0
    for comp, df in task_dict.items():
        if df.empty:
            print(f"No data for {comp}, skipping...")
            i += 1
            continue
            
        df = df.copy()
        df = df[df['model'] != 'scPermut']
        sns.set_theme(style="white")
        n_dataset = df['dataset_name'].nunique()
        f, ax = plt.subplots(1, figsize=(xsizes[i], ysizes[i]), dpi=200)
    
        metric_name = metric_names.get(met, met)
        
        df['dataset_name'] = df['dataset_name'].replace(aestetic_data_name)
        df['model'] = df['model'].replace(aestetic_model_name)
        
        sns.boxplot(x='dataset_name', y=f'{split}_{met}', hue='model', data=df, 
                    hue_order=list(colors.keys()), ax=ax, palette=sns.color_palette(list(colors.values())), 
                    orient='v', flierprops={"marker": "."})
                    
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        
        ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        ax.set_ylabel(f'{metric_name} on Test')
        ax.set_xlabel('')
        ax.set(ylim=(0.2, 1))
        ax.set_title(comp)

        # Add stat bars
        if args.stats:
            pairs = []
            unique_datasets = df['dataset_name'].dropna().unique()
            if len(unique_datasets) > 0 and len(colors.keys()) > 0:
                first_model = list(colors.keys())[0]
                for dataset in unique_datasets:
                    for k in range(1, len(colors.keys())):
                        competitor_model = list(colors.keys())[k]
                        # Verify both model distributions exist for annotation parsing
                        if not df[(df['dataset_name'] == dataset) & (df['model'] == first_model)].empty and \
                           not df[(df['dataset_name'] == dataset) & (df['model'] == competitor_model)].empty:
                            pair = ((dataset, first_model), (dataset, competitor_model))
                            pairs.append(pair)
                
                if pairs:
                    annotator = Annotator(ax, pairs, data=df, x='dataset_name', y=f'{split}_{met}', hue='model', hue_order=list(colors.keys()))
                    try:
                        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
                        annotator.apply_and_annotate()
                    except Exception as e:
                        print(f"Stats annotation err on {comp}: {e}")
    
        plt.tight_layout()
        dir_path = os.path.join(fig_dir, 'task_1', 'prediction_boxplot', f'{split}_{met}')
        check_dir(dir_path)
        output_file = os.path.join(dir_path, f'prediction_boxplot_{comp}.png')
        ax.get_figure().savefig(output_file, transparent=False)
        print(f"Saved exact replica Figure 2 format to {output_file}")
        
        output_csv = os.path.join(dir_path, f'prediction_boxplot_{comp}.csv')
        cols = ['dataset_name', 'model', f'{split}_{met}']
        if 'test_fold_nb' in df.columns: cols.append('test_fold_nb')
        df[cols].to_csv(output_csv, index=False)
        print(f"Saved underlying data for {comp} to {output_csv}")
        
        i += 1
