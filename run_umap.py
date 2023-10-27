from workflow.workflow import Workflow
from load import load_runfile
import sys

working_dir = sys.argv[1]
runfile_path = sys.argv[2]

workflow = Workflow(run_file=load_runfile(runfile_path), working_dir = working_dir)

if workflow.check_run_log() and not workflow.check_umap_log():
    workflow.load_results()
    workflow.compute_umap()
    print('umap done')

elif workflow.check_umap_log():
    print('umap already exists')
    
elif not workflow.check_run_log():
    print('run doesnt exist')

workflow.load_results()
for n_neighbors in [10,50,100, 200]:
        workflow.predict_knn_classifier(n_neighbors=n_neighbors)
workflow.predict_kmeans()
workflow.compute_leiden()
workflow.save_results()