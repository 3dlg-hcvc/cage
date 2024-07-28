import os
import sys
import torch
import numpy as np
import trimesh
from copy import deepcopy
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.motions import transform_all_parts

def _load_and_combine_plys(dir, ply_files, scale=None, z_rotate=None, translate=None):
    """
    Load and combine the ply files into one PyTorch3D mesh

    - dir: the directory of the object in which the ply files are from\n
    - ply_files: the list of ply files\n
    - scale: the scale factor to apply to the vertices\n
    - z_rotate: whether to rotate the object around the z-axis by 90 degrees\n
    - translate: the translation to apply to the vertices\n

    Return:\n
    - mesh: one PyTorch3D mesh of the combined ply files
    """

    # Combine the ply files into one
    meshes = []
    for ply_file in ply_files:
        meshes.append(trimesh.load(os.path.join(dir, ply_file), force="mesh"))
    full_part_mesh = trimesh.util.concatenate(meshes)

    # Apply the transformations
    full_part_mesh.vertices -= full_part_mesh.bounding_box.centroid
    transformation = trimesh.transformations.compose_matrix(scale=scale, angles=[0, 0, np.radians(90) if z_rotate else 0], translate=translate)
    full_part_mesh.apply_transform(transformation)
    
    # Create the PyTorch3D mesh
    mesh = Meshes(verts=torch.as_tensor(full_part_mesh.vertices, dtype=torch.float32).unsqueeze(0), 
                  faces=torch.as_tensor(full_part_mesh.faces, dtype=torch.int64).unsqueeze(0))

    return mesh

def _compute_chamfer_distance(obj1_part_points, obj2_part_points):
    """
    Compute the chamfer distance between the two set of points representing the two objects

    - obj1_part_points: the set of points representing the first object\n
    - obj2_part_points: the set of points representing the second object\n

    Return:\n
    - distance: the chamfer distance between the two objects
    """

    # Merge the points of all parts into one tensor
    obj1_part_points = obj1_part_points.reshape(-1, 3)
    obj2_part_points = obj2_part_points.reshape(-1, 3)

    # Compute the chamfer distance between the two objects
    with torch.no_grad():
        obj1_part_points = obj1_part_points.cuda()
        obj2_part_points = obj2_part_points.cuda()
        distance, _ = chamfer_distance(obj1_part_points[None, :], obj2_part_points[None, :], batch_reduction=None)
        distance = distance.item()
    
    return distance

def ID(gen_obj_dict, gen_obj_retrieved_parts, gt_obj_dict, gt_obj_path, num_states=5, num_samples=2048):
    """
    Compute the ID metric\n
    This metric is the average chamfer distance between the two objects over a number of articulation states\n

    - gen_obj_dict: the generated object dictionary\n
    - gen_obj_retrieved_parts: the list of retrieved parts of the generated object\n
    - gt_obj_dict: the ground truth object dictionary\n
    - gt_obj_path: the path to the ground truth object\n
    - num_states (optional): the number of articulation states to compute the metric\n
    - num_samples (optional): the number of samples to use\n

    Return:\n
    - score: the metric score, which is the overall average chamfer distance over the states\n
        - The score is in the range of [0, inf), lower is better
    """

    if "diffuse_tree" not in gt_obj_dict: # Rename for compatibility
        gt_obj_dict["diffuse_tree"] = gt_obj_dict.pop("arti_tree")

    # Get the number of parts of the two objects
    gen_obj_num_parts = len(gen_obj_dict["diffuse_tree"])
    gt_obj_num_parts = len(gt_obj_dict["diffuse_tree"])

    # Get the paths of the ply files of the two objects
    gen_obj_part_ply_paths = [{"dir": gen_obj_retrieved_parts[i]["dir"], "files": gen_obj_retrieved_parts[i]["files"]} 
                              for i in range(gen_obj_num_parts)]
    gt_obj_part_ply_paths = [{"dir": gt_obj_path, "files": gt_obj_dict["diffuse_tree"][i]["objs"]} 
                             for i in range(gt_obj_num_parts)]
    
    # Load the ply files of the two objects and sample points from them
    gen_obj_part_points = torch.zeros((gen_obj_num_parts, num_samples, 3))
    for i in range(gen_obj_num_parts):
        part_mesh = _load_and_combine_plys(gen_obj_part_ply_paths[i]["dir"], gen_obj_part_ply_paths[i]["files"], 
                                           scale=gen_obj_retrieved_parts[i]["scale_factor"], 
                                           z_rotate=gen_obj_retrieved_parts[i]["z_rotate_90"], 
                                           translate=gen_obj_dict["diffuse_tree"][i]["aabb"]["center"])
        gen_obj_part_points[i] = sample_points_from_meshes(part_mesh, num_samples=num_samples).squeeze(0)

    gt_obj_part_points = torch.zeros((gt_obj_num_parts, num_samples, 3))
    for i in range(gt_obj_num_parts):
        part_mesh = _load_and_combine_plys(gt_obj_part_ply_paths[i]["dir"], gt_obj_part_ply_paths[i]["files"], 
                                           translate=gt_obj_dict["diffuse_tree"][i]["aabb"]["center"])
        gt_obj_part_points[i] = sample_points_from_meshes(part_mesh, num_samples=num_samples).squeeze(0)
    
    original_gen_obj_part_points = deepcopy(gen_obj_part_points)
    original_gt_obj_part_points = deepcopy(gt_obj_part_points)

    chamfer_distances = np.zeros(num_states)
    joint_states = np.linspace(0, 1, num_states)
    for state_idx, state in enumerate(joint_states):

        # Reset the part point clouds
        gen_obj_part_points = deepcopy(original_gen_obj_part_points)
        gt_obj_part_points = deepcopy(original_gt_obj_part_points)

        # Transform the part point clouds to the current state using the joints
        transform_all_parts(gen_obj_part_points.numpy(), gen_obj_dict, state, dry_run=False)
        transform_all_parts(gt_obj_part_points.numpy(), gt_obj_dict, state, dry_run=False)

        # Compute the chamfer distance between the two objects
        gen_to_gt_distance = _compute_chamfer_distance(gen_obj_part_points, gt_obj_part_points)
        gt_to_gen_distance = _compute_chamfer_distance(gt_obj_part_points, gen_obj_part_points)

        # Store the chamfer distance
        chamfer_distances[state_idx] = (gen_to_gt_distance + gt_to_gen_distance) * 0.5

    # Compute the ID
    score = np.mean(chamfer_distances)

    return score