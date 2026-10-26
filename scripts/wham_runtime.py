"""Select the CUDA libraries belonging to the WHAM Python environment."""
import os
from pathlib import Path
import sys


def runtime_environment(prefix, inherited=None):
    environment = dict(os.environ if inherited is None else inherited)
    prefix = Path(prefix)
    libraries = []
    # cu118 wheels may bundle CUDA/cuDNN inside torch/lib rather than nvidia/*/lib.
    for site in sorted((prefix/'lib').glob('python*/site-packages')):
        torch_lib = site/'torch/lib'
        if torch_lib.is_dir():
            libraries.append(str(torch_lib))
        libraries.extend(str(p) for p in sorted((site/'nvidia').glob('*/lib')) if p.is_dir())
    if libraries:
        inherited_paths = environment.get('LD_LIBRARY_PATH','').split(':')
        environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(libraries+[p for p in inherited_paths if p]))
    return environment


def ensure_runtime():
    """The loader needs LD_LIBRARY_PATH before the Python process starts."""
    environment = runtime_environment(sys.prefix)
    if environment.get('LD_LIBRARY_PATH','') != os.environ.get('LD_LIBRARY_PATH',''):
        os.execve(sys.executable,[sys.executable,*sys.argv],environment)
