from workflow.workflow import Workflow
from load import load_runfile
import sys

working_dir = sys.argv[1]
runfile_path = sys.argv[2] 

workflow = Workflow(run_file=load_runfile(runfile_path), working_dir = working_dir)

if workflow.check_run_log():
    if not workflow.check_predict_log():
        workflow.compute_prediction_only()
        workflow.save_results()
        print('prediction done')

    elif workflow.check_predict_log():
        print('prediction already exists')
        
elif not workflow.check_run_log():
    print('Run does not exist')