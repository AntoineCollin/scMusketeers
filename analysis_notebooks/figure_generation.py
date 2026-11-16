import os
import argparse
import logging

# python analysis_notebooks/figure_generation.py --process_csv --task1
# python analysis_notebooks/figure_generation.py --process_csv --task2_new
# python analysis_notebooks/figure_generation.py --task1
# python analysis_notebooks/figure_generation.py --task2
# python analysis_notebooks/figure_generation.py --task2_new


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from figure_scripts.csv_processing import csv_process
from figure_scripts.generate_figure_task1 import produce_fig_2
from figure_scripts.generate_figure_task1_New import produce_fig_task1_New
from figure_scripts.generate_figure_task2 import produce_fig_3
from figure_scripts.generate_figure_task2_New import produce_fig_task2_New
from figure_scripts.generate_heatmap import produce_heatmap

def main():
    working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(working_dir, 'experiment_script/results')

    checkpoint_dir = os.path.join(working_dir, 'analysis_notebooks', 'csv_batches')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'metrics_checkpoint.csv')

    parser = argparse.ArgumentParser(description="Generate Figures from SC Musketeers Benchmark metrics")
    parser.add_argument("--process_csv", action="store_true", help="Process raw result metrics into unified checkpoint CSVs per task")
    parser.add_argument("--task1", action="store_true", help="Load the Task 1 checkpoint CSV and produce Task 1 figure")
    parser.add_argument("--task1_new", action="store_true", help="Load the Task 1 New checkpoint CSV and produce Task 1 New figure (new datasets)")
    parser.add_argument("--task2", action="store_true", help="Load the Task 2 checkpoint CSV and produce Task 2 figure")
    parser.add_argument("--task2_new", action="store_true", help="Load the Task 2 New checkpoint CSV and produce Task 2 New figure (new datasets)")
    parser.add_argument("--heatmap", action="store_true", help="Produce the Task 1 model-rank heatmap (median balanced accuracy)")
    parser.add_argument("--all", action="store_true", help="Run everything from start to finish")
    parser.add_argument("--run_stats", action="store_true", help="Enable Wilcoxon statistical annotations on figures")

    args = parser.parse_args()

    run_csv = args.process_csv or args.all
    run_task1 = args.task1 or args.all
    run_task1_new = args.task1_new or args.all
    run_task2 = args.task2 or args.all
    run_task2_new = args.task2_new or args.all
    run_heatmap = args.heatmap or args.all

    if not (run_csv or run_task1 or run_task1_new or run_task2 or run_task2_new or run_heatmap):
        logger.info("No explicit flags passed, defaulting to '--all'.")
        run_csv = True
        run_task1 = True
        run_task1_new = True
        run_task2 = True
        run_task2_new = True
        run_heatmap = True

    if run_csv:
        # task1 group and "all others" (task2_New group) are processed independently
        # so each can be refreshed without reprocessing the other.
        if run_task1:
            logger.info("--- Starting CSV Processing: task1 ---")
            csv_process(results_dir, checkpoint_path, task_filter=["1"])
        if run_task1_new or run_task2 or run_task2_new:
            logger.info("--- Starting CSV Processing: task2_New (all others) ---")
            csv_process(results_dir, checkpoint_path, task_filter=["1_New", "2", "2_New"])
        if not (run_task1 or run_task1_new or run_task2 or run_task2_new):
            logger.info("--- Starting CSV Processing: all tasks ---")
            csv_process(results_dir, checkpoint_path)

    if run_task1:
        logger.info("--- Starting Task 1 Figure Generation ---")
        produce_fig_2(checkpoint_path, working_dir, run_stats=args.run_stats)

    if run_task1_new:
        logger.info("--- Starting Task 1 New Figure Generation ---")
        produce_fig_task1_New(checkpoint_path, working_dir, run_stats=args.run_stats)

    if run_task2:
        logger.info("--- Starting Task 2 Figure Generation ---")
        produce_fig_3(checkpoint_path, working_dir, run_stats=args.run_stats)

    if run_task2_new:
        logger.info("--- Starting Task 2 New Figure Generation ---")
        produce_fig_task2_New(checkpoint_path, working_dir, run_stats=args.run_stats)

    if run_heatmap:
        logger.info("--- Starting Task 1 New Rank Heatmap Generation ---")
        produce_heatmap(checkpoint_path, working_dir, task="1_New")
        logger.info("--- Starting Task 2 New Rank Heatmap Generation ---")
        produce_heatmap(checkpoint_path, working_dir, task="2_New")

if __name__ == "__main__":
    main()
