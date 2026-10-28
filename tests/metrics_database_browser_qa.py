"""Compatibility entry point for current v2 database browser regression checks."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('biomechanics_browser_qa.py')),run_name='__main__')
