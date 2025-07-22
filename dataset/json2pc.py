import os
import glob
import json
import numpy as np
import random
import h5py
from joblib import Parallel, delayed
from trimesh.sample import sample_surface
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD, BatchCADSolid2views
from utils.pc_utils import write_ply, read_ply
from PIL import Image

DATA_ROOT = "../data"
RAW_DATA = os.path.join(DATA_ROOT, "cad_json")
RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")

N_POINTS = 8096 # 4096
WRITE_NORMAL = False
SAVE_DIR = os.path.join(DATA_ROOT, "pc_cad")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

INVALID_IDS = []


def process_one(data_ids, only_pc=False):
    real_data_ids=[]
    shapes=[]
    for data_id in data_ids:
        if data_id in INVALID_IDS:
            print("skip {}: in invalid id list".format(data_id))
            continue

        save_path = os.path.join(SAVE_DIR, data_id + ".ply")
        if os.path.exists(save_path):
            if only_pc or os.path.exists(os.path.join(SAVE_DIR, f'{data_id}_render_metadata.txt')):
                print("skip {}: file already exists".format(data_id))
                continue

        json_path = os.path.join(RAW_DATA, data_id + ".json")
        with open(json_path, "r") as fp:
            data = json.load(fp)

        try:
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            shape = create_CAD(cad_seq)
        except Exception as e:
            print("create_CAD failed:", data_id)
            continue
        try:
            out_pc = CADsolid2pc(shape, N_POINTS, data_id.split("/")[-1])
        except Exception as e:
            print("convert point cloud failed:", data_id)
            continue
            
        save_path = os.path.join(SAVE_DIR, data_id + ".ply")
        truck_dir = os.path.dirname(save_path)
        if not os.path.exists(truck_dir):
            os.makedirs(truck_dir)

        write_ply(out_pc, save_path)
        real_data_ids.append(data_id)
        shapes.append(shape)

    t=time.time()
    if len(real_data_ids)==0:
        return
    if not only_pc:
        try:
            all_out_images,metadatas=BatchCADSolid2views(shapes, [data_id.split("/")[-1] for data_id in real_data_ids])
        except Exception as e:
            print(e)
            print("convert to image failed:", data_id)
            return None
            
        print(f'Images creation: {(time.time()-t):.2f} seconds')

    if not only_pc:
        for out_images,metadata,data_id in zip(all_out_images,metadatas,real_data_ids):
            for k,img in enumerate(out_images):
                save_img_path = os.path.join(SAVE_DIR, f'{data_id}_{k:02d}.png')
                image=Image.fromarray(img)
                image.save(save_img_path)
            print(save_img_path)

            save_meta_path = os.path.join(SAVE_DIR, f'{data_id}_render_metadata.txt')
            with open(save_meta_path,'w') as f:
                f.write(metadata)


with open(RECORD_FILE, "r") as fp:
    all_data = json.load(fp)


parser = argparse.ArgumentParser()
parser.add_argument('--only_test', action="store_true", help="only convert test data")
parser.add_argument('--only_pc', action="store_true", help="generate point clouds only")
parser.add_argument('--step2', action="store_true", help="Step 2")
args = parser.parse_args()

if args.only_pc:
    if not args.only_test:
        Parallel(n_jobs=10, verbose=2)(delayed(process_one)([x],True) for x in all_data["train"])
        Parallel(n_jobs=10, verbose=2)(delayed(process_one)([x],True) for x in all_data["validation"])
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)([x],True) for x in all_data["test"])
else:
    import time
    if not args.only_test:
        deltas_t=[0 for k in range(5)]
        n=len(all_data['train'])//20
        for k in range(5000 if args.step2 else 0, n):
            x=all_data['train'][(k*20):((k+1)*20)]
            t=time.time()
            process_one(x)
            deltas_t.append(time.time()-t)
            avg_dur=sum(deltas_t[-5:])/5
            estimated_remaining=(avg_dur*(n-k))/3600
            print(f'Processed batch {k}/{n} of training set. Average time per shape: {(avg_dur/20):.2f} s. Estimated remaining time: {estimated_remaining:.2f} hours.')
        
        deltas_t=[0 for k in range(5)]
        n=len(all_data['validation'])//20
        for k in range(n):
            x=all_data['validation'][(k*20):((k+1)*20)]
            t=time.time()
            process_one(x)
            deltas_t.append(time.time()-t)
            avg_dur=sum(deltas_t[-5:])/5
            estimated_remaining=(avg_dur*(n-k))/3600
            print(f'Processed batch {k}/{n} of validation set.  Average time per shape: {(avg_dur/20):.2f} s. Estimated remaining time: {estimated_remaining:.2f} hours.')
    
    n=len(all_data['test'])//20
    deltas_t=[0 for k in range(5)]
    for k in range(n):
        t=time.time()
        x=all_data['test'][(k*20):((k+1)*20)]
        process_one(x)
        deltas_t.append(time.time()-t)
        avg_dur=sum(deltas_t[-5:])/5
        estimated_remaining=(avg_dur*(n-k))/3600
        print(f'Processed batch {k}/{n} of test set. Average time per shape: {(avg_dur/20):.2f} s. Estimated remaining time: {estimated_remaining:.2f} hours.')
