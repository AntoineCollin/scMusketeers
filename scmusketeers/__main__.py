from scmusketeers.run_musk import run_sc_musketeers
from scmusketeers.arguments.runfile import (
    get_runfile,
)

def main_entry_point():
    # Get all arguments
    run_file = get_runfile()
    #run_file = get_default_param()
    print(run_file)
    run_sc_musketeers(run_file)

if __name__ == "__main__":
    main_entry_point()
