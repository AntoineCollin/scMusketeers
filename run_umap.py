from workflow.workflow import Workflow
import sys

working_dir = sys.argv[1]
runfile_path = sys.argv[2]

workflow = Workflow(yaml_name = runfile_path, working_dir = working_dir)

if workflow.check_run_log() and not workflow.check_umap_log():
    workflow.load_results()
    workflow.compute_umap()
    print('umap done')
        
elif workflow.check_umap_log():
    print('umap already exists')
    
elif not workflow.check_run_log():
    print('run doesnt exist')