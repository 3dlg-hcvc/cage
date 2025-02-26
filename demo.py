import os
import json
import trimesh
import numpy as np
import argparse

import torch
import lightning.pytorch as pl
import datamodules
import systems
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from utils.callbacks import ConfigSnapshotCallback

from retrieval.obj_retrieval import find_obj_candidates, pick_and_rescale_parts
from utils.misc import load_config

def retrieve_meshes(obj_name, save_dir, dataset_path="./data"):
    print(f"Retrieving meshes for {obj_name}...")
    # Load the object spec json file
    with open(os.path.join(save_dir, f"{obj_name}.json"), "r") as f:
        obj_spec = json.load(f)
    
    HASHBOOK_PATH = "./retrieval/retrieval_hash_no_handles.json"
    
    # Retrieve meshes for the object
    obj_candidates = find_obj_candidates(obj_spec, dataset_path, HASHBOOK_PATH, gt_file_name="train.json")
    retrieved_mesh_specs = pick_and_rescale_parts(obj_spec, obj_candidates, dataset_path, gt_file_name="train.json")

    # ============================ Load the meshes and save them as a PLY file for each part
    mesh_dir_name = f"{obj_name}_meshes"
    os.makedirs(os.path.join(save_dir, mesh_dir_name), exist_ok=True)
    scene = trimesh.Scene()
    for i, mesh_spec in enumerate(retrieved_mesh_specs):
        part_spec = obj_spec["diffuse_tree"][i]

        # A part may be composed of multiple meshes
        part_meshes = []
        for file in mesh_spec["files"]:
            mesh = trimesh.load(os.path.join(mesh_spec["dir"], file), force="mesh")
            part_meshes.append(mesh)
        part_mesh = trimesh.util.concatenate(part_meshes)
        
        # Center the mesh at the origin
        part_mesh.vertices -= part_mesh.bounding_box.centroid

        transformation = trimesh.transformations.compose_matrix(scale=mesh_spec["scale_factor"], 
                                                                angles=[0, 0, np.radians(90) if mesh_spec["z_rotate_90"] else 0], 
                                                                translate=part_spec["aabb"]["center"])
        part_mesh.apply_transform(transformation)
        # Save the part mesh as a PLY file
        part_mesh.export(os.path.join(save_dir, mesh_dir_name, f"part_{i}.ply"))
        # Update object json
        obj_spec["diffuse_tree"][i]["plys"] = [f"{mesh_dir_name}/part_{i}.ply"]
        # Add the mesh to the scene
        scene.add_geometry(part_mesh)

    # Export the scene as a PLY file
    save_path = os.path.join(save_dir, f"{obj_name}.ply")
    scene.export(save_path)
    # Export the updated json
    with open(os.path.join(save_dir, f"{obj_name}.json"), "w") as f:
        json.dump(obj_spec, f)

def main(config, args):

    assert args.ckpt is not None

    # ============================ Run prediction on a trained model

    dm = datamodules.make(config.system.datamodule.name, config=config.system.datamodule)
    system = systems.make(config.system.name, config.system)

    logger = TensorBoardLogger(save_dir='exps', name=config.name, version=config.version)
    callbacks = [ModelCheckpoint(**config.checkpoint), LearningRateMonitor(), ModelSummary(), ConfigSnapshotCallback(config)]

    trainer = pl.Trainer(devices='auto',
                        strategy='ddp', 
                        accelerator='auto',
                        logger=logger,
                        callbacks=callbacks,
                        profiler="simple",
                        **config.trainer)

    checkpoint = torch.load(args.ckpt)
    trainer.fit_loop.load_state_dict(checkpoint['loops']['fit_loop'])
    trainer.test_loop.load_state_dict(checkpoint['loops']['test_loop'])
    trainer.predict(system, datamodule=dm, ckpt_path=args.ckpt)

    # ============================ Retreive meshes for each generated object
    obj_spec_json_dir = os.path.join('exps', config.name, config.version, "images", "predict", 'ood')
    for root, _, files in os.walk(obj_spec_json_dir, topdown=False):
        for file in files:
            if file.endswith(".json") and file.startswith("#"):
                # Retrieve the meshes for the object and save them as a PLY file
                fname = file.split(".")[0]
                retrieve_meshes(fname, root, dataset_path=args.data_root)

if __name__ == "__main__":
    '''
    This script runs prediction on a pre-trained model in an OOD mode.
    The model predicts 10 samples by default for the input graph specified in the demo_graph.json.
    For each generated object, the script retrieves the meshes for each part and records the object hierarchy in the json file.
    All the saved files can be found in the "exps/cage/demo/images/predict" directory.

    Pre-requisites:
    - Since we rely on the training data to retrieve the meshes, the data should be present in the "<project folder>/data" directory already.
    - Download the pre-trained.zip file and extract it in the "exps" directory.
    - The file structure should look like this:
        <project folder>
            |-- data
            |-- exps
                |-- cage
                    |-- demo
                        |-- checkpoints
                            |-- last.ckpt
                        |-- config
                            |-- parsed.yaml
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exps/cage/demo/config/parsed.yaml", help="path to the config file")
    parser.add_argument("--ckpt", default="exps/cage/demo/checkpoints/last.ckpt", help="path to the trained weights")
    parser.add_argument("--data_root", default="./data", help="path to the data root directory")
    parser.add_argument("--input_graphs", default="demo_graph.json", help="path to the input graph json file")
    parser.add_argument("--pred_n_samples", type=int, default=10, help="number of samples to generate for the input graph")

    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    assert os.path.exists(args.data_root), f"Data root directory not found: {args.data_root}"
    assert os.path.exists(args.ckpt), f"Checkpoint file not found: {args.ckpt}"
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.input_graphs), f"Input graph json file not found: {args.input_graphs}"

    # ----- Edit to the config to use the input graph json
    config.system.datamodule.pred_mode = "ood"              # Set to "ood" prediction mode to take the input graph json
    config.system.datamodule.input_graphs = args.input_graphs   # Set to the path of the input graph json file
    config.system.datamodule.root = args.data_root
    config.system.datamodule.pred_n_samples = args.pred_n_samples

    main(config, args)
