"""Use the CUDA libraries of the actual inherited torch, not the overlay venv.

No global shell/environment mutation. Re-exec only this Python process before
loading torch, necessary for CUDA LSTM/cuDNN when using system-site-packages.
"""
import os,sys,importlib.util
from pathlib import Path

def ensure_runtime():
 spec=importlib.util.find_spec('torch')
 if spec is None:raise RuntimeError('torch unavailable')
 site=Path(spec.origin).parent.parent;paths=[site/'torch/lib']+sorted((site/'nvidia').glob('*/lib'));paths=[str(p) for p in paths if p.is_dir()]
 environment=dict(os.environ);old=[p for p in environment.get('LD_LIBRARY_PATH','').split(':') if p];environment['LD_LIBRARY_PATH']=':'.join(dict.fromkeys(paths+old))
 if environment['LD_LIBRARY_PATH']!=os.environ.get('LD_LIBRARY_PATH',''):os.execve(sys.executable,[sys.executable,*sys.argv],environment)
