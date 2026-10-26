#!/usr/bin/env python3
"""Prepare and execute the actual WHAM/Halpe-26 pipeline in the server SSH shell."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

POSE_URL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip'


def run(arguments, **kwargs):
    print('+', ' '.join(map(str,arguments)), flush=True)
    subprocess.run(list(map(str,arguments)),check=True,**kwargs)


def extract_body_assets(archive, destination):
    """Extract only regular files/directories confined to the intended private folder."""
    destination = destination.resolve()
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            path = (destination/member.name).resolve()
            if destination not in path.parents and path != destination:
                raise ValueError('Unsafe body-model archive path')
            if not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected body-model archive link/device')
        source.extractall(destination,members=members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[2]
    parser.add_argument('--workspace',type=Path,default=workspace)
    parser.add_argument('--smpl-neutral',type=Path,default=Path.home()/'.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,default=0,help='0 processes all 813 dataset clips; 1 is a smoke run')
    parser.add_argument('--deploy',action='store_true',help='commit numeric data and push main after processing')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    # Stop before installation in a sandbox without hardware/network access.
    run(['nvidia-smi'])
    for url in ['https://github.com/yohanshin/WHAM','https://download.pytorch.org/whl/cu118/',POSE_URL,'https://drive.google.com']:
        with urllib.request.urlopen(url,timeout=20) as response:
            print('NETWORK',url,response.status,flush=True)
    if not args.smpl_neutral.is_file():
        raise SystemExit('Existing licensed SMPL neutral model missing: '+str(args.smpl_neutral)+
            '\nPass --smpl-neutral /path/to/SMPL_NEUTRAL.pkl after obtaining it from SMPLify.')
    conda = os.environ.get('CONDA_EXE') or shutil.which('conda') or str(Path.home()/'miniconda3/bin/conda')
    if not Path(conda).is_file():
        raise SystemExit('Conda executable not found; activate the server Miniconda installation first.')
    environment = workspace/'.wham-env'
    python = environment/'bin/python'
    if not python.exists():
        run([conda,'create','--yes','--prefix',environment,'python=3.9','pip'])
    marker = environment/'kbo-wham-dependencies-v1.json'
    if not marker.exists():
        run([python,'-m','pip','install','torch==2.1.2+cu118','torchvision==0.16.2+cu118',
            '--index-url','https://download.pytorch.org/whl/cu118'])
        run([python,'-m','pip','install','numpy==1.23.5','setuptools==65.5.0','wheel'])
        run([python,'-m','pip','install','chumpy==0.70','--no-build-isolation'])
        run([python,'-m','pip','install','numpy==1.23.5','scipy==1.10.1','onnxruntime-gpu==1.16.3',
            'opencv-python-headless==4.10.0.84','smplx==0.1.28','yacs==0.1.8','joblib==1.4.2',
            'einops==0.6.1','timm==0.4.9','scikit-image==0.21.0','progress==1.6',
            'loguru==0.7.2','gdown==5.2.0','PyYAML','tqdm','imageio[ffmpeg]',
            'matplotlib==3.7.5','munkres','tensorboard','omegaconf==2.3.0',
            'pytorch-lightning==2.1.3','torchmetrics==1.3.2'])
        # RTMLib can use onnxruntime-gpu; avoid installing a competing CPU ORT/OpenCV wheel.
        run([python,'-m','pip','install','rtmlib==0.0.16','--no-deps'])
        marker.write_text(json.dumps({'python':'3.9','torch':'2.1.2+cu118','version':1}))
    # The CUDA 11 libraries installed above must be visible to ONNX Runtime as well as torch.
    libraries = list((environment/'lib/python3.9/site-packages/nvidia').glob('*/lib'))
    os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str,libraries))+':'+os.environ.get('LD_LIBRARY_PATH','')
    root = workspace/'external/WHAM'
    if not root.exists():
        root.parent.mkdir(parents=True,exist_ok=True)
        run(['git','clone','--depth','1','https://github.com/yohanshin/WHAM.git',root])
    if not (root/'lib/models/wham.py').is_file():
        raise SystemExit('The existing WHAM checkout is incomplete: '+str(root))
    checkpoints = root/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    def download(identifier, destination):
        if destination.is_file() and destination.stat().st_size:
            return
        temporary = destination.with_suffix(destination.suffix+'.part')
        run([python,'-m','gdown',identifier,'-O',temporary])
        temporary.replace(destination)
    # IDs from the official WHAM fetch_demo_data.sh; no alternative 3D checkpoint.
    download('19qkI-a6xuwob9_RFNSPWf1yWErwVVlks',checkpoints/'wham_vit_bedlam_w_3dpw.pth.tar')
    download('1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y',checkpoints/'hmr2a.ckpt')
    body_root = root/'dataset/body_models'
    if not all((body_root/name).is_file() for name in
               ['J_regressor_wham.npy','J_regressor_h36m.npy','J_regressor_feet.npy','smpl_mean_params.npz']):
        archive = checkpoints/'body_models.tar.gz'
        download('1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ',archive)
        (root/'dataset').mkdir(exist_ok=True)
        extract_body_assets(archive,root/'dataset')
    neutral = body_root/'smpl/SMPL_NEUTRAL.pkl'
    neutral.parent.mkdir(parents=True,exist_ok=True)
    if not neutral.exists():
        neutral.symlink_to(args.smpl_neutral.resolve())
    pose = workspace/'models/rtmpose_halpe26_m.onnx'
    if not pose.exists():
        pose.parent.mkdir(parents=True,exist_ok=True)
        archive = checkpoints/'rtmpose_halpe26_m.zip'
        urllib.request.urlretrieve(POSE_URL,archive)
        with zipfile.ZipFile(archive) as source:
            models = [name for name in source.namelist() if name.endswith('.onnx')]
            if len(models)!=1:
                raise ValueError('Expected one ONNX model in the official Halpe-26 archive')
            temporary = pose.with_suffix('.part')
            with source.open(models[0]) as incoming,temporary.open('wb') as outgoing:
                shutil.copyfileobj(incoming,outgoing)
            temporary.replace(pose)
    script = Path(__file__).resolve().with_name('wham_pipeline.py')
    command = [python,script,'--dataset',workspace/'dataset','--wham-root',root,
               '--pose-model',pose,'--device',args.device]
    run(command+['--check'])
    if args.prepare_only:
        return 0
    command += ['--publish','--limit',str(args.limit)]
    if args.deploy:
        command.append('--deploy')
    run(command)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
