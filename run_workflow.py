from workflow.workflow import Workflow
import sys

working_dir = sys.argv[1]
print(sys.argv)
runfile_path = sys.argv[2]

workflow = Workflow(yaml_name = runfile_path, working_dir = working_dir)

if not workflow.check_run_log():
    workflow.make_experiment()
    print(workflow.predict_done)
    workflow.save_results()
    print('training done')

elif workflow.check_run_log():
    print('run already exists')