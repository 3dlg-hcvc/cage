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

def main(config, args):

    assert args.ckpt is not None

    # ============================ Run prediction on a trained model

    dm = datamodules.make(config.system.datamodule.name, config=config.system.datamodule)
    system = systems.make(config.system.name, config.system, args.ckpt)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=config.name, version=config.version)
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

    # ============================ Retreive meshes for a generated object

    DATASET_PATH = "./data"
    HASHBOOK_PATH = "./retrieval/retrieval_hash_no_handles.json"

    # Find the first object spec json file
    obj_spec_json_dir = os.path.join(args.log_dir, config.name, config.version, "images", "predict")
    for root, _, files in os.walk(obj_spec_json_dir, topdown=False):
        for file in files:
            if file == "#0.json":
                obj_spec_json_path = os.path.join(root, file)
                break
    assert obj_spec_json_path is not None

    # Load the object spec json file
    with open(obj_spec_json_path, "r") as f:
        obj_spec = json.load(f)

    # Retrieve meshes for the object
    obj_candidates = find_obj_candidates(obj_spec, DATASET_PATH, HASHBOOK_PATH, gt_file_name="train.json")
    retrieved_mesh_specs = pick_and_rescale_parts(obj_spec, obj_candidates, DATASET_PATH, gt_file_name="train.json")

    # ============================ Load the meshes and save them as a GLB file

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

        # Rotate the mesh 90 degrees about the z-axis if specified
        if mesh_spec["z_rotate_90"]:
            part_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1]))

        # Scale and position the mesh
        transform_matrix = trimesh.transformations.scale_and_translate(mesh_spec["scale_factor"], part_spec["aabb"]["center"])
        part_mesh.apply_transform(transform_matrix)

        # Add the mesh to the scene
        scene.add_geometry(part_mesh)

    # Export the scene as a GLB file
    scene.export("demo.glb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exps/cage/head_4_layer_8/config/parsed.yaml")
    parser.add_argument("--log_dir", type=str, default="exps")
    parser.add_argument("--ckpt", default="exps/cage/head_4_layer_8/checkpoints/last.ckpt", help="path to the trained weights")
    parser.add_argument("--graph_json", default="demo_graph.json", help="path to the graph json file")

    args, extras = parser.parse_known_args()

    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    # ----- Edit to the config to use the input graph json
    config.system.datamodule.pred_mode = "ood"              # Set to "ood" to use input graph json
    config.system.datamodule.graph_json = args.graph_json   # Set to the path of the graph json file

    main(config, args)