"""scMusketeers package initialisation.

The heavy ML stack (TensorFlow / oneDNN / absl / CUDA) is imported eagerly by
the submodules below. It writes a number of native banners straight to the C
stderr while it loads. We configure the environment and a file-descriptor
redirection here, *before* those submodules are imported, so the CLI and any
``import scmusketeers`` stay quiet.
"""

import contextlib
import os
import sys
import warnings

# Must be set before TensorFlow is imported (below) to take effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TF C++ INFO/WARNING/ERROR
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
# Disable oneDNN custom ops: it is a CPU optimisation (training runs on GPU), so
# turning it off has negligible cost while removing the non-deterministic-results
# banner at its source and making numerics reproducible across machines.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Deprecation noise raised at import time by dependencies (e.g. celltypist).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirect C-level stderr (fd 2) to /dev/null during heavy imports.

    TensorFlow / oneDNN / absl / CUDA write banners straight to fd 2 from C++
    *before* absl logging is initialised, so they ignore TF_CPP_MIN_LOG_LEVEL
    and the Python logging configuration. Redirecting the file descriptor is
    the only reliable way to hide them. Real import errors still surface: they
    raise a Python exception whose traceback is printed after stderr is
    restored.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


from . import arguments

# These pull in the whole TensorFlow/Keras stack; mute its native import-time
# banners.
with _suppress_native_stderr():
    from . import tools
    from . import transfer
    from . import run_musk

from . import __main__
