from workflow.hyperparameters import Workflow
from scpermut.workflow import runfile


if __name__ == '__main__':
    
    run_file = runfile.get_runfile()
    print(run_file.dataset_name, run_file.class_key, run_file.batch_key)

    workflow = Workflow(run_file=run_file, working_dir=run_file.working_dir)
    workflow.start_neptune_log()
    workflow.process_dataset()
    workflow.split_train_test()
    mcc = workflow.make_experiment()
    workflow.stop_neptune_log()

