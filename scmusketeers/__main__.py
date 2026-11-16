import logging
from importlib.metadata import version

from scmusketeers.arguments.runfile import get_runfile
from scmusketeers.run_musk import run_sc_musketeers

logger = logging.getLogger("Sc-Musketeers")


def _setup_logging():
    """Attach a dedicated handler to the "Sc-Musketeers" logger.

    We cannot rely on logging.basicConfig() here: importing the heavy ML
    stack (TensorFlow/absl) installs a handler on the root logger before
    main_entry_point() runs, which turns basicConfig() into a silent no-op.
    Instead we configure our own logger directly and stop propagation so the
    format is always applied and messages are not emitted twice.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("|--- %(levelname)-8s    %(message)s")
        )
        logger.addHandler(handler)
    logger.propagate = False


def main_entry_point():
    # Set up logging
    _setup_logging()

    # Get all arguments first: --version / --help exit here before we log
    # anything, keeping their output clean.
    run_file = get_runfile()

    # Set logger level
    if run_file.debug:
        logger.setLevel(getattr(logging, "DEBUG"))
    else:
        logger.setLevel(getattr(logging, "INFO"))

    logger.info(f"Sc-Musketeers {version('sc-musketeers')} started")
    logger.debug(f"Program arguments: {run_file}")
    
    #run_file = get_default_param()
    run_sc_musketeers(run_file)

if __name__ == "__main__":
    main_entry_point()
