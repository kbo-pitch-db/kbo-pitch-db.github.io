#!/usr/bin/env python3
"""Collect only current, atomically completed contact fits from private indexes."""
import argparse,csv
from pathlib import Path
from package_scene import records
from package_pitch_scene import checked_record


def collect(dataset):
    base=records(dataset/'motion_wham.csv');previous=records(dataset/'motion_wham_scene.csv');completed={}
    paths=[dataset/'motion_wham_pitch_scene.csv',*sorted(dataset.glob('motion_wham_pitch_scene.worker*.csv'))]
    for path in paths:
        for cid,row in records(path).items():
            if cid not in base:continue
            try:checked_record(dataset,cid,row,base[cid],previous[cid])
            except (ValueError,KeyError,FileNotFoundError):continue
            completed[cid]=row
    if not completed:raise ValueError('No current completed fits')
    tmp=dataset/'motion_wham_pitch_scene.csv.tmp'
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset_clip_id','pose_3d_scene','scene_quality','scene_signature','method','fit_accepted']);writer.writeheader();writer.writerows(completed[k] for k in sorted(completed))
    tmp.replace(dataset/'motion_wham_pitch_scene.csv');print('Collected current completed fits:',len(completed));return completed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('../dataset'));a=p.parse_args();collect(a.dataset)
